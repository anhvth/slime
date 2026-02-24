# Privileged Distillation: Prompt Construction & Token Alignment Report

## 1. How Teacher and Student Prompts Look

### Student Prompt (NO privileged context)

The student sees the **original** tokenized sequence, unchanged:

```
[prompt_token_ids]  [response_token_ids]
 ^-- P tokens        ^-- R tokens
```

The student model runs normal forward on this during Megatron training. Its logits for the response portion are extracted by `get_responses()` in `slime/backends/megatron_utils/loss.py` (line 34).

### Teacher Prompt (WITH privileged context)

Built in `_build_teacher_input_ids_and_start_len()` at `opd_topk_reward_plugin.py` lines 192–224:

```
[prompt_token_ids]  [privileged_context_tokens]  [response_token_ids]
 ^-- P tokens        ^-- X extra tokens           ^-- R tokens (same as student)
```

Concretely, the privileged context is injected as:

```python
suffix = f"\n[PRIVILEGED_CONTEXT]\n{privileged_context}\n[/PRIVILEGED_CONTEXT]\n"
privileged_ids = tokenizer.encode(suffix, add_special_tokens=False)
scored_prompt_ids = prompt_ids + privileged_ids
return scored_prompt_ids + response_ids, len(scored_prompt_ids)
# returns: (full_input_ids, logprob_start_len)
```

**Example** (conceptual, with token IDs replaced by words):

| Part | Student sees | Teacher sees |
|------|-------------|-------------|
| Prompt | `<system>You are helpful</system><user>Solve x²=4</user>` | `<system>You are helpful</system><user>Solve x²=4</user>` |
| Privileged | *(nothing)* | `\n[PRIVILEGED_CONTEXT]\nThe answer is x=±2 because...\n[/PRIVILEGED_CONTEXT]\n` |
| Response | `<think>Let me solve...</think>x=±2` | `<think>Let me solve...</think>x=±2` |

### Where does the privileged context come from?

`_resolve_privileged_context()` (line 64) checks:
1. `sample.metadata["privileged_context"]` — from the dataset's `metadata` column
2. Falls back to `sample.label` if `opd_privileged_fallback_label=1`

The privileged context is **pre-generated** by `build_privileged_dataset.py`, which calls a teacher LLM to produce a `<distill_context>...</distill_context>` block per sample and stores it in the `metadata.privileged_context` field.

---

## 2. How Teacher Scores the Student's Response

### The SGLang API call

In `_build_teacher_payload()` (`opd_topk_reward_plugin.py` line 226):

```python
def _build_teacher_payload(args, sample, topk):
    input_ids, logprob_start_len = _build_teacher_input_ids_and_start_len(args, sample)
    return {
        "input_ids": input_ids,                     # [P + X + R] tokens
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,     # = P + X  ← KEY!
        "top_logprobs_num": topk,
    }
```

**Critical parameter: `logprob_start_len = P + X`**

This tells SGLang: "only return logprobs starting at position `P + X`", which is exactly where the **response tokens begin**. SGLang will:
- Process all `P + X + R` tokens (teacher sees privileged context in its KV cache)
- Return `input_top_logprobs` only for positions `>= P + X` → exactly `R` entries
- Each entry contains top-K `(logprob, token_id)` pairs

### Parsing the teacher output

In `extract_topk_from_reward()` → `parse_input_top_logprobs_rows()` (`opd_topk_parser.py` line 70):

```python
selected_rows = rows[-response_length:]  # Take LAST response_length rows
```

Since SGLang returns exactly `R` rows (due to `logprob_start_len = P + X`), `rows[-R:]` == `rows` — all R response rows are selected. The output shape is `[R, K]` for both logprobs and token_ids.

---

## 3. Token Alignment Verification: Is Shifting Handled Correctly?

### The alignment chain (with evidence)

Here is the complete data flow, showing dimensions at each step:

#### Step 1: Teacher scoring (per-sample, async HTTP)

```
Teacher input:  [P + X + R] tokens
logprob_start:  P + X
SGLang returns:  R rows of top-K logprobs
Parser output:   teacher_topk_logprobs  [R, K]
                 teacher_topk_token_ids [R, K]
```

**Evidence**: `opd_topk_reward_plugin.py` line 214:
```python
scored_prompt_ids = prompt_ids + privileged_ids
return scored_prompt_ids + response_ids, len(scored_prompt_ids)
#                                        ^^^^^^^^^^^^^^^^^^^^
#                                        logprob_start_len = P + X
```

#### Step 2: Student forward pass (Megatron training)

```
Student input:  [P + R] tokens  (NO privileged context)
get_responses() yields per sample:
  logits_chunk:  [R, V]  (response logits)
  tokens_chunk:  [R]     (response tokens)
```

**Evidence**: `loss.py` lines 93–95 (cp_size=1 path):
```python
end += total_length
start = end - response_length
logits_chunk = logits[start - 1 : end - 1]   # [R, V] — the "-1" is the standard LM shift
tokens_chunk = tokens[-response_length:]       # [R]
```

The `start - 1 : end - 1` shift is the standard autoregressive shift: logit at position `t-1` predicts token at position `t`. This gives `R` logits aligned to the `R` response tokens.

#### Step 3: Loss computation (per-sample, in loss plugin)

```python
# opd_topk_loss_plugin.py, distill_topk_custom_loss(), line ~134
teacher_lp = teacher_topk_logprobs[i]   # [R, K]
teacher_ids = teacher_topk_token_ids[i]  # [R, K]

if teacher_lp.size(0) != logits_chunk.size(0):   # R == R ?
    raise ValueError("Teacher response span does not match model response span ...")

student_lp = _gather_selected_logprobs_tp(logits_chunk, teacher_ids)  # [R, K]
```

**The dimensionality assertion** at line ~142 ensures teacher and student tensors have the same number of response positions. If any misalignment existed, this would raise immediately.

### What each position means

For response position `j` (0-indexed, `j ∈ [0, R)`):

| | Teacher | Student |
|---|---------|---------|
| **Predicting** | `response[j]` | `response[j]` |
| **Conditioned on** | `prompt + privileged + response[0..j-1]` | `prompt + response[0..j-1]` |
| **Tensor location** | `teacher_topk_logprobs[j, :]` | `logits_chunk[j, :]` |

Both predict the **same token** at **the same position** — the only difference is the teacher has additional privileged context in its conditioning. This is exactly the desired behavior for privileged information distillation.

### Why the extra privileged tokens do NOT cause misalignment

The key insight: **privileged tokens are never part of the logprob output**. The mechanism:

1. `logprob_start_len = P + X` skips all prompt + privileged tokens in SGLang's output
2. The parser takes `-response_length:` rows, which are exactly the response positions
3. The student's `get_responses()` also yields exactly `response_length` logits
4. Both are `R`-length, aligned position-by-position on the same response tokens

There is **no off-by-one** because:
- SGLang's `input_top_logprobs[i]` = logprob of predicting `input_ids[i]` given `input_ids[0:i]`
- For `i ≥ P+X`, this means predicting `response[i-(P+X)]` given all prior tokens
- Megatron's `logits[start-1 : end-1]` applies the standard LM shift, yielding logits that predict `response[0], response[1], ..., response[R-1]`

### Edge cases handled

1. **No privileged context available**: `_resolve_privileged_context()` returns `None` → `logprob_start_len = P` (no extra offset) → same as non-privileged distillation
2. **Mixed privileged/non-privileged in the same batch**: Each sample is processed independently; the dimensionality check `teacher_lp.size(0) != logits_chunk.size(0)` validates per-sample
3. **Padding in top-K**: If teacher returns fewer than K entries for a position, `parse_input_top_logprobs_rows()` pads with `logprob=-1e9, token_id=0` — these have near-zero probability after softmax and don't affect the loss

---

## 4. Summary

| Aspect | Status | Evidence |
|--------|--------|----------|
| Teacher has privileged context, student does not | ✅ Correct | `_build_teacher_input_ids_and_start_len()` injects privileged tokens only into teacher input |
| Privileged tokens excluded from logprob output | ✅ Correct | `logprob_start_len = len(prompt) + len(privileged)` skips them |
| Teacher and student aligned on response tokens | ✅ Correct | Both yield exactly `response_length` entries; validated by shape assertion |
| Autoregressive shift handled correctly | ✅ Correct | SGLang's `input_top_logprobs[i]` predicts `token[i]`; Megatron's `logits[t-1]` predicts `token[t]` — both aligned |
| Per-sample independence | ✅ Correct | Each sample builds its own `logprob_start_len`; no cross-sample contamination |
| JSD computation correctness | ✅ Correct | Unit tests verify symmetry, non-negativity, zero for identical distributions |

**Conclusion: The token alignment is handled correctly.** The `logprob_start_len` mechanism ensures the extra privileged tokens in the teacher's input are completely invisible in the logprob output, producing an exact position-by-position match with the student's response logits.
