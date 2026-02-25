from __future__ import annotations

import logging
import time
from argparse import Namespace

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_responses

from opd_topk_parser import TOPK_PAD_LOGPROB

try:
    from opd_debug_dump import dump_topk_debug_update
except Exception:  # pragma: no cover - keep training resilient if plugin path is unavailable
    dump_topk_debug_update = None


logger = logging.getLogger(__name__)
_TOPK_PAD_THRESHOLD = TOPK_PAD_LOGPROB / 10.0


def compute_topk_renormalized_jsd(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute top-k renormalized JSD on the last dimension."""
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "Teacher/student top-k log-prob shape mismatch: "
            f"{tuple(teacher_log_probs.shape)} vs {tuple(student_log_probs.shape)}"
        )
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"JSD beta must be in [0,1], got {beta}.")

    log_t_norm = teacher_log_probs - torch.logsumexp(teacher_log_probs, dim=-1, keepdim=True)
    log_s_norm = student_log_probs - torch.logsumexp(student_log_probs, dim=-1, keepdim=True)

    probs_t = log_t_norm.exp()
    probs_s = log_s_norm.exp()

    probs_m = beta * probs_t + (1.0 - beta) * probs_s
    log_m = probs_m.clamp_min(eps).log()

    kl_t_m = (probs_t * (log_t_norm - log_m)).sum(dim=-1)
    kl_s_m = (probs_s * (log_s_norm - log_m)).sum(dim=-1)
    return beta * kl_t_m + (1.0 - beta) * kl_s_m


def _gather_selected_logprobs_tp(logits_chunk: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather logprobs for arbitrary global token ids from TP-sharded logits."""
    if logits_chunk.ndim != 2:
        raise ValueError(f"Expected logits_chunk [R, V_local], got shape={tuple(logits_chunk.shape)}")
    if token_ids.ndim != 2:
        raise ValueError(f"Expected token_ids [R, K], got shape={tuple(token_ids.shape)}")
    if logits_chunk.size(0) != token_ids.size(0):
        raise ValueError(
            f"Row mismatch between logits_chunk and token_ids: {tuple(logits_chunk.shape)} vs {tuple(token_ids.shape)}"
        )
    if logits_chunk.size(0) == 0:
        return torch.empty(token_ids.shape, device=logits_chunk.device, dtype=logits_chunk.dtype)

    logits_chunk = logits_chunk.to(torch.float32)
    token_ids = token_ids.to(device=logits_chunk.device, dtype=torch.long)

    tp_size = mpu.get_tensor_model_parallel_world_size()
    vocab_size_local = logits_chunk.size(-1)
    global_vocab_size = vocab_size_local * tp_size

    if token_ids.numel() > 0:
        token_id_min = int(token_ids.min().item())
        token_id_max = int(token_ids.max().item())
        if token_id_min < 0 or token_id_max >= global_vocab_size:
            raise ValueError(
                f"Token ids out of range: min={token_id_min}, max={token_id_max}, global_vocab_size={global_vocab_size}"
            )

    if tp_size == 1:
        logsumexp = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
        selected_logits = logits_chunk.gather(dim=-1, index=token_ids)
        return selected_logits - logsumexp

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_start = tp_rank * vocab_size_local
    vocab_end = vocab_start + vocab_size_local

    local_max = logits_chunk.max(dim=-1, keepdim=True).values
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)

    local_exp_sum = (logits_chunk - local_max).exp().sum(dim=-1, keepdim=True)
    dist.all_reduce(local_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
    logsumexp = local_exp_sum.log() + local_max

    in_local_vocab = (token_ids >= vocab_start) & (token_ids < vocab_end)
    local_indices = (token_ids - vocab_start).masked_fill(~in_local_vocab, 0)
    gathered_local = logits_chunk.gather(dim=-1, index=local_indices)
    selected_logits = torch.where(in_local_vocab, gathered_local, torch.zeros_like(gathered_local))
    dist.all_reduce(selected_logits, op=dist.ReduceOp.SUM, group=tp_group)

    return selected_logits - logsumexp


def _get_required_batch_key(batch: dict, key: str):
    value = batch.get(key)
    if value is None:
        raise ValueError(f"Missing required batch key for top-k distillation: {key}")
    return value


def _compute_forward_reverse_terms(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_lp_norm = teacher_log_probs - torch.logsumexp(teacher_log_probs, dim=-1, keepdim=True)
    student_lp_norm = student_log_probs - torch.logsumexp(student_log_probs, dim=-1, keepdim=True)

    teacher_probs = teacher_lp_norm.exp()
    student_probs = student_lp_norm.exp()

    forward_kl = (teacher_probs * (teacher_lp_norm - student_lp_norm)).sum(dim=-1)
    reverse_kl = (student_probs * (student_lp_norm - teacher_lp_norm)).sum(dim=-1)
    return forward_kl, reverse_kl


def _compute_forward_reverse_terms_unbalanced(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()
    forward_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    return forward_kl, reverse_kl


def compute_cross_tokenizer_group_losses(
    *,
    teacher_log_probs: torch.Tensor,
    student_anchor_log_probs: torch.Tensor,
    student_actual_log_probs: torch.Tensor,
    group_lengths: torch.Tensor,
    group_valid_mask: torch.Tensor,
    mode: str,
    mixed_weight: float,
    jsd_beta: float,
) -> dict[str, torch.Tensor | int]:
    """Compute per-group distillation terms for cross-tokenizer mode (vectorized).

    Args:
        teacher_log_probs: [R, K] merged teacher log-probs.
        student_anchor_log_probs: [R, K] student log-probs on teacher support at anchor rows.
        student_actual_log_probs: [R] student log-probs of actual response tokens.
        group_lengths: [R] anchor row stores group span length, non-anchor rows store 0.
        group_valid_mask: [R] anchor row validity, non-anchor rows store 0.
    """
    if teacher_log_probs.shape != student_anchor_log_probs.shape:
        raise ValueError(
            "teacher/student anchor log-prob shape mismatch in cross-tokenizer mode: "
            f"{tuple(teacher_log_probs.shape)} vs {tuple(student_anchor_log_probs.shape)}"
        )
    if teacher_log_probs.ndim != 2:
        raise ValueError(f"Expected [R,K] tensors, got {tuple(teacher_log_probs.shape)}")

    response_len, topk = teacher_log_probs.shape
    if student_actual_log_probs.ndim != 1 or student_actual_log_probs.size(0) != response_len:
        raise ValueError(
            "student_actual_log_probs must be [R], got "
            f"shape={tuple(student_actual_log_probs.shape)}, expected R={response_len}"
        )
    if group_lengths.ndim != 1 or group_lengths.size(0) != response_len:
        raise ValueError(f"group_lengths must be [R], got shape={tuple(group_lengths.shape)}")
    if group_valid_mask.ndim != 1 or group_valid_mask.size(0) != response_len:
        raise ValueError(f"group_valid_mask must be [R], got shape={tuple(group_valid_mask.shape)}")

    device = teacher_log_probs.device
    dtype = teacher_log_probs.dtype

    # Find all anchor positions (group_lengths > 0)
    anchor_mask = group_lengths > 0
    num_anchors = int(anchor_mask.sum().item())
    total_support = num_anchors * topk

    empty_result = {
        "forward": torch.empty((0,), device=device, dtype=dtype),
        "reverse": torch.empty((0,), device=device, dtype=dtype),
        "jsd": torch.empty((0,), device=device, dtype=dtype),
        "forward_debug": torch.zeros((response_len,), device=device, dtype=dtype),
        "reverse_debug": torch.zeros((response_len,), device=device, dtype=dtype),
        "jsd_debug": (
            torch.zeros((response_len,), device=device, dtype=dtype)
            if mode == "jsd"
            else torch.empty((0,), device=device, dtype=dtype)
        ),
        "valid_groups": 0,
        "skipped_groups": 0,
        "mapped_support": 0,
        "total_support": total_support,
    }

    if num_anchors == 0:
        return empty_result

    anchor_indices = anchor_mask.nonzero(as_tuple=True)[0]  # [num_anchors]
    anchor_lens = group_lengths[anchor_indices]  # [num_anchors]
    anchor_ends = anchor_indices + anchor_lens  # [num_anchors]
    anchor_valid = group_valid_mask[anchor_indices] > 0
    anchor_in_bounds = anchor_ends <= response_len
    usable = anchor_valid & anchor_in_bounds

    skipped_groups = num_anchors - int(usable.sum().item())

    if not usable.any():
        empty_result["skipped_groups"] = skipped_groups
        empty_result["total_support"] = total_support
        return empty_result

    usable_idx = anchor_indices[usable]  # [N]
    usable_lens = anchor_lens[usable]  # [N]
    usable_ends = anchor_ends[usable]  # [N]

    # Compute continuation logprobs via prefix sum (vectorized)
    prefix_sum = torch.zeros(response_len + 1, device=device, dtype=dtype)
    prefix_sum[1:] = torch.cumsum(student_actual_log_probs, dim=0)
    continuation = prefix_sum[usable_ends] - prefix_sum[usable_idx + 1]
    continuation = torch.where(usable_lens > 1, continuation, torch.zeros_like(continuation))

    # Batch gather teacher and student logprobs for all usable anchors
    teacher_lp = teacher_log_probs[usable_idx]  # [N, K]
    student_lp = student_anchor_log_probs[usable_idx] + continuation.unsqueeze(-1)  # [N, K]

    # Batch compute forward/reverse KL
    forward_all, reverse_all = _compute_forward_reverse_terms_unbalanced(teacher_lp, student_lp)  # [N], [N]

    # JSD if needed
    if mode == "jsd":
        jsd_all = compute_topk_renormalized_jsd(teacher_lp, student_lp, beta=jsd_beta)  # [N]
    else:
        jsd_all = torch.empty((0,), device=device, dtype=dtype)

    # Support coverage (vectorized)
    mapped_support = int((teacher_lp > _TOPK_PAD_THRESHOLD).sum().item())
    valid_groups = int(usable_idx.size(0))

    # Debug per-position tensors
    forward_debug = torch.zeros((response_len,), device=device, dtype=dtype)
    reverse_debug = torch.zeros((response_len,), device=device, dtype=dtype)
    forward_debug[usable_idx] = forward_all.detach()
    reverse_debug[usable_idx] = reverse_all.detach()

    if mode == "jsd":
        jsd_debug = torch.zeros((response_len,), device=device, dtype=dtype)
        jsd_debug[usable_idx] = jsd_all.detach()
    else:
        jsd_debug = torch.empty((0,), device=device, dtype=dtype)

    return {
        "forward": forward_all,
        "reverse": reverse_all,
        "jsd": jsd_all,
        "forward_debug": forward_debug,
        "reverse_debug": reverse_debug,
        "jsd_debug": jsd_debug,
        "valid_groups": valid_groups,
        "skipped_groups": skipped_groups,
        "mapped_support": mapped_support,
        "total_support": total_support,
    }


def _resolve_distill_terms(
    *,
    mode: str,
    mixed_weight: float,
    forward_all: torch.Tensor,
    reverse_all: torch.Tensor,
    jsd_all: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    if mode == "fkl":
        return forward_all, 1.0
    if mode == "mixed":
        return mixed_weight * forward_all + (1.0 - mixed_weight) * reverse_all, 2.0

    if jsd_all is None:
        raise ValueError("JSD mode requires jsd_all tensor.")
    return jsd_all, 3.0


def _extract_response_token_ids(sample_tokens: torch.Tensor | list[int], response_len: int) -> torch.Tensor:
    if isinstance(sample_tokens, torch.Tensor):
        if sample_tokens.ndim != 1:
            sample_tokens = sample_tokens.reshape(-1)
        return sample_tokens[-response_len:].to(dtype=torch.long)

    return torch.tensor(sample_tokens[-response_len:], dtype=torch.long)


def distill_topk_custom_loss(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean,
):
    """Custom loss for top-k distillation with FKL / mixed-KL / JSD."""
    distill_total_start_time = time.perf_counter()
    distill_time_student_gather_s = 0.0
    distill_time_cross_group_s = 0.0
    distill_time_debug_dump_s = 0.0

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size != 1:
        raise ValueError(
            "Top-k distillation v1 supports only context_parallel_size=1. "
            f"Got context_parallel_size={cp_size}."
        )

    mode = str(getattr(args, "distill_loss_mode", "fkl")).lower()
    if mode not in {"fkl", "mixed", "jsd"}:
        raise ValueError(f"distill_topk_custom_loss supports modes ['fkl', 'mixed', 'jsd'], got {mode!r}.")

    mixed_weight = float(getattr(args, "opd_mixed_kl_weight", 0.5))
    if not (0.0 <= mixed_weight <= 1.0):
        raise ValueError(f"opd_mixed_kl_weight must be in [0,1], got {mixed_weight}.")

    jsd_beta = float(getattr(args, "opd_jsd_beta", 0.5))
    if not (0.0 <= jsd_beta <= 1.0):
        raise ValueError(f"opd_jsd_beta must be in [0,1], got {jsd_beta}.")

    distill_coef = float(getattr(args, "opd_distill_coef", 1.0))

    teacher_topk_logprobs: list[torch.Tensor] = _get_required_batch_key(batch, "teacher_topk_logprobs")
    teacher_topk_token_ids: list[torch.Tensor] = _get_required_batch_key(batch, "teacher_topk_token_ids")

    cross_tokenizer = bool(getattr(args, "opd_cross_tokenizer_enable", False))
    teacher_topk_group_lengths = batch.get("teacher_topk_group_lengths")
    teacher_topk_group_valid_mask = batch.get("teacher_topk_group_valid_mask")
    if cross_tokenizer:
        if teacher_topk_group_lengths is None or teacher_topk_group_valid_mask is None:
            raise ValueError(
                "Cross-tokenizer distillation requires batch keys "
                "teacher_topk_group_lengths and teacher_topk_group_valid_mask."
            )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    per_forward: list[torch.Tensor] = []
    per_reverse: list[torch.Tensor] = []
    per_jsd: list[torch.Tensor] | None = [] if mode == "jsd" else None
    per_sample_forward_means: list[torch.Tensor] = []
    per_sample_reverse_means: list[torch.Tensor] = []
    per_sample_jsd_means: list[torch.Tensor] | None = [] if mode == "jsd" else None
    topk_values: list[int] = []

    valid_groups_total = 0
    skipped_groups_total = 0
    mapped_support_total = 0.0
    support_den_total = 0.0

    debug_records: list[dict] = []
    sample_indices = batch.get("sample_indices")
    teacher_input_ids = batch.get("teacher_input_ids")
    teacher_logprob_start_len = batch.get("teacher_logprob_start_len")

    for i, (logits_chunk, _) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=batch.get("max_seq_lens", None),
        )
    ):
        teacher_lp = teacher_topk_logprobs[i].to(device=logits_chunk.device, dtype=torch.float32)
        teacher_ids = teacher_topk_token_ids[i].to(device=logits_chunk.device, dtype=torch.long)

        if teacher_lp.ndim != 2 or teacher_ids.ndim != 2:
            raise ValueError(
                "Expected per-sample teacher tensors with shape [response_len, topk], "
                f"got logprobs={tuple(teacher_lp.shape)}, ids={tuple(teacher_ids.shape)}"
            )
        if teacher_lp.shape != teacher_ids.shape:
            raise ValueError(
                f"Teacher top-k tensor shape mismatch: logprobs={tuple(teacher_lp.shape)} vs ids={tuple(teacher_ids.shape)}"
            )
        if teacher_lp.size(0) != logits_chunk.size(0):
            raise ValueError(
                "Teacher response span does not match model response span for sample "
                f"{i}: teacher_len={teacher_lp.size(0)}, model_len={logits_chunk.size(0)}"
            )

        topk_values.append(int(teacher_lp.size(1)))

        if cross_tokenizer:
            group_lengths = teacher_topk_group_lengths[i].to(device=logits_chunk.device, dtype=torch.long)
            group_valid_mask = teacher_topk_group_valid_mask[i].to(device=logits_chunk.device, dtype=torch.long)

            gather_start_time = time.perf_counter()
            student_anchor_lp = _gather_selected_logprobs_tp(logits_chunk, teacher_ids)
            student_debug_lp = student_anchor_lp

            response_token_ids = _extract_response_token_ids(batch["unconcat_tokens"][i], int(teacher_lp.size(0))).to(
                device=logits_chunk.device
            )
            student_actual_lp = _gather_selected_logprobs_tp(logits_chunk, response_token_ids.unsqueeze(-1)).squeeze(-1)
            distill_time_student_gather_s += time.perf_counter() - gather_start_time

            cross_group_start_time = time.perf_counter()
            group_res = compute_cross_tokenizer_group_losses(
                teacher_log_probs=teacher_lp,
                student_anchor_log_probs=student_anchor_lp,
                student_actual_log_probs=student_actual_lp,
                group_lengths=group_lengths,
                group_valid_mask=group_valid_mask,
                mode=mode,
                mixed_weight=mixed_weight,
                jsd_beta=jsd_beta,
            )
            distill_time_cross_group_s += time.perf_counter() - cross_group_start_time

            forward_kl = group_res["forward"]
            reverse_kl = group_res["reverse"]
            jsd = group_res["jsd"]
            if forward_kl.numel() > 0:
                per_forward.append(forward_kl)
                per_reverse.append(reverse_kl)
                if per_jsd is not None:
                    per_jsd.append(jsd)
                per_sample_forward_means.append(forward_kl.mean())
                per_sample_reverse_means.append(reverse_kl.mean())
                if per_sample_jsd_means is not None:
                    per_sample_jsd_means.append(jsd.mean())

            valid_groups_total += int(group_res["valid_groups"])
            skipped_groups_total += int(group_res["skipped_groups"])
            mapped_support_total += float(group_res["mapped_support"])
            support_den_total += float(group_res["total_support"])

            forward_debug = group_res["forward_debug"]
            reverse_debug = group_res["reverse_debug"]
            jsd_debug = group_res["jsd_debug"] if per_jsd is not None else None
        else:
            gather_start_time = time.perf_counter()
            student_lp = _gather_selected_logprobs_tp(logits_chunk, teacher_ids)
            distill_time_student_gather_s += time.perf_counter() - gather_start_time
            student_debug_lp = student_lp
            forward_kl, reverse_kl = _compute_forward_reverse_terms(teacher_lp, student_lp)
            jsd = compute_topk_renormalized_jsd(teacher_lp, student_lp, beta=jsd_beta) if per_jsd is not None else None

            per_forward.append(forward_kl)
            per_reverse.append(reverse_kl)
            if per_jsd is not None:
                assert jsd is not None
                per_jsd.append(jsd)

            valid_groups_total += int(forward_kl.numel())
            mapped_support_total += float(forward_kl.numel() * teacher_lp.size(1))
            support_den_total += float(forward_kl.numel() * teacher_lp.size(1))

            forward_debug = forward_kl
            reverse_debug = reverse_kl
            jsd_debug = jsd

        sample_tokens = batch["unconcat_tokens"][i]
        response_len = int(teacher_lp.size(0))
        total_len = int(sample_tokens.size(0)) if hasattr(sample_tokens, "size") else len(sample_tokens)
        response_start = int(total_len - response_len)
        debug_record = {
            "sample_index": sample_indices[i] if sample_indices is not None else -1,
            "microbatch_sample_index": i,
            "student_input_ids": sample_tokens,
            "response_start": response_start,
            "response_length": response_len,
            "teacher_input_ids": teacher_input_ids[i] if teacher_input_ids is not None else None,
            "teacher_logprob_start_len": (
                teacher_logprob_start_len[i] if teacher_logprob_start_len is not None else None
            ),
            "teacher_topk_logprobs": teacher_lp,
            "teacher_topk_token_ids": teacher_ids,
            "student_topk_logprobs": student_debug_lp,
            "forward_kl": forward_debug,
            "reverse_kl": reverse_debug,
            "jsd": jsd_debug,
            "cross_tokenizer": int(cross_tokenizer),
        }
        if cross_tokenizer:
            debug_record["teacher_topk_group_lengths"] = group_lengths
            debug_record["teacher_topk_group_valid_mask"] = group_valid_mask
        debug_records.append(debug_record)

    if dump_topk_debug_update is not None and debug_records:
        debug_dump_start_time = time.perf_counter()
        try:
            dump_topk_debug_update(args, mode=mode, records=debug_records)
        except Exception:
            logger.warning("Failed to save top-k distillation debug dump.", exc_info=True)
        finally:
            distill_time_debug_dump_s += time.perf_counter() - debug_dump_start_time

    if not per_forward:
        zero = 0.0 * logits.sum()
        empty_mode_code = {"fkl": 1.0, "mixed": 2.0, "jsd": 3.0}[mode]
        distill_time_total_s = time.perf_counter() - distill_total_start_time
        return (
            zero,
            {
                "loss": zero.clone().detach(),
                "distill_kl": zero.clone().detach(),
                "distill_kl_forward": zero.clone().detach(),
                "distill_kl_reverse": zero.clone().detach(),
                "distill_jsd": zero.clone().detach(),
                "distill_mode": torch.tensor(empty_mode_code, device=logits.device),
                "distill_topk": torch.tensor(0.0, device=logits.device),
                "distill_valid_groups": torch.tensor(0.0, device=logits.device),
                "distill_skipped_groups": torch.tensor(float(skipped_groups_total), device=logits.device),
                "distill_support_coverage": torch.tensor(0.0, device=logits.device),
                "distill_time_total_s": torch.tensor(float(distill_time_total_s), device=logits.device),
                "distill_time_student_gather_s": torch.tensor(float(distill_time_student_gather_s), device=logits.device),
                "distill_time_cross_group_s": torch.tensor(float(distill_time_cross_group_s), device=logits.device),
                "distill_time_debug_dump_s": torch.tensor(float(distill_time_debug_dump_s), device=logits.device),
            },
        )

    forward_all = torch.cat(per_forward, dim=0)
    reverse_all = torch.cat(per_reverse, dim=0)
    jsd_all = torch.cat(per_jsd, dim=0) if per_jsd is not None else None

    distill_all, mode_code = _resolve_distill_terms(
        mode=mode,
        mixed_weight=mixed_weight,
        forward_all=forward_all,
        reverse_all=reverse_all,
        jsd_all=jsd_all,
    )

    if cross_tokenizer:
        if per_sample_forward_means:
            sample_forward_all = torch.stack(per_sample_forward_means)
            sample_reverse_all = torch.stack(per_sample_reverse_means)
            if per_sample_jsd_means is not None:
                sample_jsd_all = torch.stack(per_sample_jsd_means)
            else:
                sample_jsd_all = None

            sample_distill_all, _ = _resolve_distill_terms(
                mode=mode,
                mixed_weight=mixed_weight,
                forward_all=sample_forward_all,
                reverse_all=sample_reverse_all,
                jsd_all=sample_jsd_all,
            )
            distill_kl = sample_distill_all.sum()
            forward_kl = sample_forward_all.sum()
            reverse_kl = sample_reverse_all.sum()
            if sample_jsd_all is not None and sample_jsd_all.numel() > 0:
                jsd_kl = sample_jsd_all.sum()
            else:
                jsd_kl = torch.zeros((), device=logits.device, dtype=distill_kl.dtype)
        else:
            distill_kl = torch.zeros((), device=logits.device, dtype=forward_all.dtype)
            forward_kl = distill_kl
            reverse_kl = distill_kl
            jsd_kl = torch.zeros((), device=logits.device, dtype=distill_kl.dtype)
    else:
        distill_kl = sum_of_sample_mean(distill_all)
        forward_kl = sum_of_sample_mean(forward_all)
        reverse_kl = sum_of_sample_mean(reverse_all)
        if jsd_all is not None:
            jsd_kl = sum_of_sample_mean(jsd_all)
        else:
            jsd_kl = torch.zeros((), device=logits.device, dtype=distill_kl.dtype)

    loss = distill_coef * distill_kl
    if distill_all.numel() == 0:
        loss = loss + 0.0 * logits.sum()

    if len(set(topk_values)) != 1:
        raise ValueError(f"Inconsistent top-k across samples in one micro-batch: {topk_values}")

    support_coverage = mapped_support_total / support_den_total if support_den_total > 0 else 0.0
    distill_time_total_s = time.perf_counter() - distill_total_start_time

    return (
        loss,
        {
            "loss": loss.clone().detach(),
            "distill_kl": distill_kl.clone().detach(),
            "distill_kl_forward": forward_kl.clone().detach(),
            "distill_kl_reverse": reverse_kl.clone().detach(),
            "distill_jsd": jsd_kl.clone().detach(),
            "distill_mode": torch.tensor(mode_code, device=logits.device),
            "distill_topk": torch.tensor(float(topk_values[0]), device=logits.device),
            "distill_valid_groups": torch.tensor(float(valid_groups_total), device=logits.device),
            "distill_skipped_groups": torch.tensor(float(skipped_groups_total), device=logits.device),
            "distill_support_coverage": torch.tensor(float(support_coverage), device=logits.device),
            "distill_time_total_s": torch.tensor(float(distill_time_total_s), device=logits.device),
            "distill_time_student_gather_s": torch.tensor(float(distill_time_student_gather_s), device=logits.device),
            "distill_time_cross_group_s": torch.tensor(float(distill_time_cross_group_s), device=logits.device),
            "distill_time_debug_dump_s": torch.tensor(float(distill_time_debug_dump_s), device=logits.device),
        },
    )
