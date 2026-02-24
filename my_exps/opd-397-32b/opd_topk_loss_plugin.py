from __future__ import annotations

import logging
from argparse import Namespace

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_responses

try:
    from opd_debug_dump import dump_topk_debug_update
except Exception:  # pragma: no cover - keep training resilient if plugin path is unavailable
    dump_topk_debug_update = None


logger = logging.getLogger(__name__)


def compute_topk_renormalized_jsd(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute top-k renormalized JSD on the last dimension.

    Args:
        teacher_log_probs: [..., K] teacher log-probs on top-k support.
        student_log_probs: [..., K] student log-probs on the same support.
        beta: Mixture weight in [0,1].
        eps: Numerical floor for mixture probabilities.

    Returns:
        [...]-shaped tensor with per-row JSD values.
    """
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
    """Gather logprobs for arbitrary global token ids from TP-sharded logits.

    Args:
        logits_chunk: [R, V_local] float32 tensor on current TP rank.
        token_ids: [R, K] global token ids.

    Returns:
        [R, K] log probabilities in global vocab space.
    """
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

    # Compute global log-sum-exp denominator.
    local_max = logits_chunk.max(dim=-1, keepdim=True).values
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)

    local_exp_sum = (logits_chunk - local_max).exp().sum(dim=-1, keepdim=True)
    dist.all_reduce(local_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
    logsumexp = local_exp_sum.log() + local_max

    # Gather selected global logits using one masked local gather + TP all-reduce.
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


def distill_topk_custom_loss(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean,
):
    """Custom loss for top-k distillation with FKL / mixed-KL / JSD."""
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

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    per_token_forward: list[torch.Tensor] = []
    per_token_reverse: list[torch.Tensor] = []
    per_token_jsd: list[torch.Tensor] | None = [] if mode == "jsd" else None
    topk_values: list[int] = []
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

        student_lp = _gather_selected_logprobs_tp(logits_chunk, teacher_ids)

        # Renormalize both distributions to the top-k support so that the
        # backward gradient through logsumexp becomes zero-sum.  Without this,
        # FKL creates a dense shift gradient across all V vocab logits (the
        # "mass_topk · softmax" term), inflating grad-norm by ~O(sqrt(V/K)).
        teacher_lp_norm = teacher_lp - torch.logsumexp(teacher_lp, dim=-1, keepdim=True)
        student_lp_norm = student_lp - torch.logsumexp(student_lp, dim=-1, keepdim=True)

        teacher_probs = teacher_lp_norm.exp()
        student_probs = student_lp_norm.exp()

        forward_kl = (teacher_probs * (teacher_lp_norm - student_lp_norm)).sum(dim=-1)
        reverse_kl = (student_probs * (student_lp_norm - teacher_lp_norm)).sum(dim=-1)
        jsd = None
        per_token_forward.append(forward_kl)
        per_token_reverse.append(reverse_kl)
        if per_token_jsd is not None:
            jsd = compute_topk_renormalized_jsd(teacher_lp, student_lp, beta=jsd_beta)
            per_token_jsd.append(jsd)

        sample_tokens = batch["unconcat_tokens"][i]
        response_len = int(teacher_lp.size(0))
        total_len = int(sample_tokens.size(0)) if hasattr(sample_tokens, "size") else len(sample_tokens)
        response_start = int(total_len - response_len)
        debug_records.append(
            {
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
                "student_topk_logprobs": student_lp,
                "forward_kl": forward_kl,
                "reverse_kl": reverse_kl,
                "jsd": jsd,
            }
        )

    if dump_topk_debug_update is not None and debug_records:
        try:
            dump_topk_debug_update(args, mode=mode, records=debug_records)
        except Exception:
            logger.warning("Failed to save top-k distillation debug dump.", exc_info=True)

    if not per_token_forward:
        zero = 0.0 * logits.sum()
        empty_mode_code = {"fkl": 1.0, "mixed": 2.0, "jsd": 3.0}[mode]
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
            },
        )

    forward_all = torch.cat(per_token_forward, dim=0)
    reverse_all = torch.cat(per_token_reverse, dim=0)
    jsd_all = torch.cat(per_token_jsd, dim=0) if per_token_jsd is not None else None

    if mode == "fkl":
        distill_all = forward_all
        mode_code = 1.0
    elif mode == "mixed":
        distill_all = mixed_weight * forward_all + (1.0 - mixed_weight) * reverse_all
        mode_code = 2.0
    else:
        assert jsd_all is not None
        distill_all = jsd_all
        mode_code = 3.0

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
        },
    )
