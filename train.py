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

    start = time.perf_counter()
    heartbeat_count = 0
    while True:
        ready, _ = ray.wait([obj_ref], timeout=heartbeat)
        if ready:
            result = ray.get(ready[0])
            logger.info(
                "[DEBUG] %s finished in %.2fs",
                stage_name,
                time.perf_counter() - start,
            )
            return result

        heartbeat_count += 1
        logger.info(
            "[DEBUG] Waiting for %s (elapsed=%.2fs, heartbeat=%s)",
            stage_name,
            time.perf_counter() - start,
            heartbeat_count,
        )


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        _ray_get_with_heartbeat(
            rollout_manager.onload_weights.remote(),
            stage_name="rollout_manager.onload_weights()",
        )

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        _ray_get_with_heartbeat(
            rollout_manager.check_weights.remote(action="compare"),
            stage_name="rollout_manager.check_weights(compare)",
        )

    if args.offload_rollout:
        _ray_get_with_heartbeat(
            rollout_manager.onload_kv.remote(),
            stage_name="rollout_manager.onload_kv()",
        )

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        _ray_get_with_heartbeat(
            rollout_manager.eval.remote(rollout_id=0),
            stage_name="rollout_manager.eval(rollout_id=0)",
        )

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
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

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            _ray_get_with_heartbeat(
                rollout_manager.eval.remote(rollout_id),
                stage_name=f"rollout_manager.eval(rollout_id={rollout_id})",
            )

        rollout_data_ref = _ray_get_with_heartbeat(
            rollout_manager.generate.remote(rollout_id),
            stage_name=f"rollout_manager.generate(rollout_id={rollout_id})",
        )

        if args.offload_rollout:
            _ray_get_with_heartbeat(
                rollout_manager.offload.remote(),
                stage_name=f"rollout_manager.offload(rollout_id={rollout_id})",
            )

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                actor_train_handle = actor_model.async_train(rollout_id, rollout_data_ref)
                _ray_get_with_heartbeat(
                    actor_train_handle,
                    stage_name=f"actor_model.async_train(rollout_id={rollout_id})",
                )
            _ray_get_with_heartbeat(
                critic_train_handle,
                stage_name=f"critic_model.async_train(rollout_id={rollout_id})",
            )
        else:
            actor_train_handle = actor_model.async_train(rollout_id, rollout_data_ref)
            _ray_get_with_heartbeat(
                actor_train_handle,
                stage_name=f"actor_model.async_train(rollout_id={rollout_id})",
            )

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            _ray_get_with_heartbeat(
                rollout_manager.onload_weights.remote(),
                stage_name=f"rollout_manager.onload_weights(rollout_id={rollout_id})",
            )
        actor_model.update_weights()
        if args.offload_rollout:
            _ray_get_with_heartbeat(
                rollout_manager.onload_kv.remote(),
                stage_name=f"rollout_manager.onload_kv(rollout_id={rollout_id})",
            )

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
