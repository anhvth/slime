import asyncio
import concurrent.futures
import logging
import os
import time

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _get_debug_heartbeat_seconds() -> float:
    raw_val = os.environ.get("SLIME_DEBUG_HEARTBEAT_SECONDS", "10")
    try:
        seconds = float(raw_val)
    except ValueError:
        logger.warning(
            "[DEBUG] Invalid SLIME_DEBUG_HEARTBEAT_SECONDS=%r. Falling back to 10.0s.",
            raw_val,
        )
        return 10.0
    return max(0.0, seconds)


def _ray_get_with_heartbeat(obj_ref, *, stage_name: str):
    heartbeat = _get_debug_heartbeat_seconds()
    if heartbeat <= 0:
        return ray.get(obj_ref)

    # Handle both single ObjectRef and list of ObjectRefs
    if isinstance(obj_ref, list):
        refs = list(obj_ref)
    else:
        refs = [obj_ref]

    start = time.perf_counter()
    heartbeat_count = 0
    while True:
        ready, refs_remaining = ray.wait(refs, num_returns=len(refs), timeout=heartbeat)
        if len(ready) == len(refs):
            result = ray.get(ready)
            logger.info(
                "[DEBUG] %s finished in %.2fs",
                stage_name,
                time.perf_counter() - start,
            )
            return result if isinstance(obj_ref, list) else result[0]

        refs = ready + refs_remaining  # keep waiting on all
        heartbeat_count += 1
        logger.info(
            "[DEBUG] Waiting for %s (elapsed=%.2fs, heartbeat=%s, ready=%d/%d)",
            stage_name,
            time.perf_counter() - start,
            heartbeat_count,
            len(ready),
            len(ready) + len(refs_remaining),
        )


def _is_cancelled_rollout_error(exc: BaseException) -> bool:
    if isinstance(exc, (concurrent.futures.CancelledError, asyncio.CancelledError)):
        return True
    if isinstance(exc, ray.exceptions.RayTaskError):
        cause = getattr(exc, "cause", None)
        if cause is not None:
            return _is_cancelled_rollout_error(cause)
        return "CancelledError" in str(exc)
    return False


def _get_rollout_data_with_retry(args, rollout_manager, rollout_data_future, rollout_id):
    max_retries = args.async_rollout_cancel_retry_times
    total_attempts = max_retries + 1

    for attempt_idx in range(total_attempts):
        attempt = attempt_idx + 1
        try:
            return _ray_get_with_heartbeat(
                rollout_data_future,
                stage_name=f"rollout_manager.generate(rollout_id={rollout_id}, attempt={attempt}/{total_attempts})",
            )
        except Exception as exc:
            if not _is_cancelled_rollout_error(exc):
                raise

            if attempt > max_retries:
                raise RuntimeError(
                    f"rollout_id={rollout_id} failed after {total_attempts} attempts due to cancelled generation"
                ) from exc

            logger.warning(
                "rollout_id=%s cancelled during async rollout get (attempt %s/%s): %s: %s",
                rollout_id,
                attempt,
                total_attempts,
                type(exc).__name__,
                exc,
            )

            if args.async_rollout_cancel_recover_engines:
                try:
                    ray.get(rollout_manager.recover_rollout_engines.remote())
                    logger.warning("Recovered rollout engines before retrying rollout_id=%s", rollout_id)
                except Exception:
                    logger.exception("Failed to recover rollout engines for rollout_id=%s", rollout_id)

            backoff_seconds = args.async_rollout_cancel_retry_backoff_base_seconds * (2**attempt_idx)
            logger.warning("Retrying rollout_id=%s after %.2fs backoff", rollout_id, backoff_seconds)
            time.sleep(backoff_seconds)
            rollout_data_future = rollout_manager.generate.remote(rollout_id)

    raise RuntimeError(f"Unreachable retry state for rollout_id={rollout_id}")


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    rollout_data_next_rollout_id = args.start_rollout_id
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = _get_rollout_data_with_retry(
                args,
                rollout_manager,
                rollout_data_next_future,
                rollout_data_next_rollout_id,
            )

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_rollout_id = rollout_id + 1
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                actor_train_handle = actor_model.async_train(rollout_id, rollout_data_curr_ref)
                _ray_get_with_heartbeat(
                    actor_train_handle,
                    stage_name=f"actor_model.async_train(rollout_id={rollout_id})",
                )
            _ray_get_with_heartbeat(
                critic_train_handle,
                stage_name=f"critic_model.async_train(rollout_id={rollout_id})",
            )
        else:
            actor_train_handle = actor_model.async_train(rollout_id, rollout_data_curr_ref)
            _ray_get_with_heartbeat(
                actor_train_handle,
                stage_name=f"actor_model.async_train(rollout_id={rollout_id})",
            )

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                _ray_get_with_heartbeat(
                    rollout_manager.save.remote(rollout_id),
                    stage_name=f"rollout_manager.save(rollout_id={rollout_id})",
                )

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (
                _get_rollout_data_with_retry(args, rollout_manager, rollout_data_next_future, rollout_data_next_rollout_id)
                if rollout_data_next_future is not None
                else None
            )
            rollout_data_next_future = None
            rollout_data_next_rollout_id = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            _ray_get_with_heartbeat(
                rollout_manager.eval.remote(rollout_id),
                stage_name=f"rollout_manager.eval(rollout_id={rollout_id})",
            )

    _ray_get_with_heartbeat(
        rollout_manager.dispose.remote(),
        stage_name="rollout_manager.dispose()",
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
