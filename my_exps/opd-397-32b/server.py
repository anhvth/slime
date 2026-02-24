"""
Single-script Ray Serve teacher deployment.

Usage:
    python server.py                  # production (397B model, 16 replicas)
    python server.py --debug          # debug (Qwen3-4B, 1 replica, 1 GPU)
    python server.py --num-replicas 4 # override replica count

Architecture:
    Client -> http://head-node:8000/teacher -> Ray Serve gateway
           -> TeacherWorker replicas (load-balanced)
           -> local sglang subprocess at 127.0.0.1:LOCAL_PORT
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Hardcoded paths — edit these to match your environment
# =============================================================================
TEACHER_PYTHON = Path("/home/anhvth8/projects/slime/my_exps/opd-397-32b/teacher-qwen-35/.venv/bin/python")
FAST_SGLANG_BIN = Path("/home/anhvth8/dotfiles/mybins/fast_sglang")
MODEL_HOME = Path(os.environ.get("MODEL_HOME", Path.home() / "ckpt/hf_models/Qwen"))

PROD_MODEL_DIR = MODEL_HOME / "Qwen3.5-397B-A17B-FP8"
PROD_MODEL_ID = "Qwen/Qwen3.5-397B-A17B-FP8"
DEBUG_MODEL_DIR = MODEL_HOME / "Qwen3-4B"

# SGLang server settings
LOCAL_HOST = "0.0.0.0"
TP = 8
CHUNKED_PREFILL_SIZE = 4096
MEM_FRACTION_STATIC = 0.8

# Ray Serve settings
GPUS_PER_REPLICA = 8
NUM_REPLICAS = 1
ROUTE_PREFIX = "/teacher"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch teacher SGLang servers via Ray Serve")
    p.add_argument("--debug", action="store_true", help="Use small debug model with 1 replica / 1 GPU")
    p.add_argument("--num-replicas", type=int, default=None, help="Override number of replicas")
    p.add_argument("--tp", type=int, default=None, help="Override tensor parallelism")
    p.add_argument("--context-length", type=int, default=None, help="Override context length")
    p.add_argument("--clean-vram", action="store_true", help="Kill existing sglang/fast_sglang processes before starting")
    p.add_argument("--ray-address", type=str, default=None, help="Ray dashboard address (e.g. http://head:8265). Auto-detected if unset.")
    return p.parse_args()


def resolve_ray_address(override: str | None) -> str:
    """Resolve the Ray Job API address (dashboard)."""
    if override:
        return override.rstrip("/")
    # Try RAY_JOB_ADDRESS env var
    env_addr = os.environ.get("RAY_JOB_ADDRESS", "").strip()
    if env_addr:
        return env_addr.rstrip("/")
    # Auto-detect from `ray job list`
    try:
        import re
        out = subprocess.run(["ray", "job", "list"], capture_output=True, text=True, timeout=10)
        combined = out.stdout + out.stderr
        # Strip ANSI codes
        combined = re.sub(r'\x1B\[[0-9;]*[a-zA-Z]', '', combined)
        m = re.search(r'https?://[^\s]+', combined)
        if m:
            return m.group(0).rstrip("/")
    except Exception:
        pass
    # Fallback
    return f"http://{os.environ.get('RAY_DASHBOARD_HOST', '127.0.0.1')}:{os.environ.get('RAY_DASHBOARD_PORT', '8265')}"


def kill_existing_sglang() -> None:
    import time
    for pattern in ("sglang.launch_server", "fast_sglang"):
        ret = subprocess.run(["pkill", "-f", pattern], capture_output=True)
        if ret.returncode == 0:
            print(f"[clean-vram] killed processes matching '{pattern}'")
    print("[clean-vram] waiting 3s for processes to exit...")
    time.sleep(3)


def resolve_model_path(debug: bool) -> str:
    if debug:
        if not DEBUG_MODEL_DIR.is_dir():
            raise FileNotFoundError(f"Debug model not found: {DEBUG_MODEL_DIR}")
        return str(DEBUG_MODEL_DIR)
    if PROD_MODEL_DIR.is_dir():
        return str(PROD_MODEL_DIR)
    return PROD_MODEL_ID


def find_free_port() -> int:
    """Bind to port 0 and let the OS assign an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_sglang_cmd(model_path: str, tp: int, port: int, context_length: int | None) -> list[str]:
    """Build the command to launch an SGLang server."""
    if not TEACHER_PYTHON.is_file():
        raise FileNotFoundError(f"Teacher python not found: {TEACHER_PYTHON}")

    base_args = [
        "--host", LOCAL_HOST,
        "--port", str(port),
        "--tp", str(tp),
        "--chunked-prefill-size", str(CHUNKED_PREFILL_SIZE),
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
    ]
    if context_length is not None:
        base_args += ["--context-length", str(context_length)]

    if FAST_SGLANG_BIN.is_file():
        return [str(TEACHER_PYTHON), str(FAST_SGLANG_BIN), model_path] + base_args

    return [str(TEACHER_PYTHON), "-m", "sglang.launch_server", "--model-path", model_path] + base_args


def _deploy(model_path: str, tp: int, gpus: int, num_replicas: int, context_length: int | None) -> None:
    """Actually deploy Ray Serve. Only called inside a Ray job on the head node."""
    import ray
    from ray import serve
    import httpx
    from starlette.responses import Response

    ray.init(address="auto")
    serve.start(detached=True)

    @serve.deployment(
        name="TeacherWorker",
        num_replicas=num_replicas,
        ray_actor_options={"num_gpus": gpus},
    )
    class TeacherWorker:
        def __init__(self):
            self.port = find_free_port()
            self.local_base = f"http://127.0.0.1:{self.port}"
            self.cmd = build_sglang_cmd(model_path, tp, self.port, context_length)
            self.client = httpx.AsyncClient(timeout=None)
            self.proc: subprocess.Popen | None = None
            print(f"TeacherWorker using port {self.port}")
            print(f"Command: {' '.join(self.cmd)}")
            self._start()

        def _start(self) -> None:
            env = {**os.environ, "PYTHONPATH": ""}
            self.proc = subprocess.Popen(self.cmd, env=env)
            print(f"Started SGLang subprocess PID={self.proc.pid} on port {self.port}")

        def _ensure_alive(self) -> None:
            if self.proc is None or self.proc.poll() is not None:
                print("SGLang subprocess dead — restarting")
                self._start()

        async def __call__(self, request):
            self._ensure_alive()

            path = request.url.path or "/"
            if path.startswith(ROUTE_PREFIX):
                path = path[len(ROUTE_PREFIX):] or "/"
            if not path.startswith("/"):
                path = f"/{path}"
            url = f"{self.local_base}{path}"
            if request.url.query:
                url = f"{url}?{request.url.query}"

            hop_by_hop = {"host", "content-length", "connection", "accept-encoding"}
            headers = {k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop}
            body = await request.body()

            resp = await self.client.request(request.method, url, headers=headers, content=body)

            resp_headers = {}
            ct = resp.headers.get("content-type")
            if ct:
                resp_headers["content-type"] = ct
            return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)

    app = TeacherWorker.bind()
    serve.run(app, name="teacher-gateway", route_prefix=ROUTE_PREFIX)
    print(f"Ray Serve deployment complete — gateway at http://<head-node>:8000{ROUTE_PREFIX}")


def main() -> None:
    args = parse_args()

    debug = args.debug
    tp = args.tp or (1 if debug else TP)
    gpus = 1 if debug else GPUS_PER_REPLICA
    num_replicas = args.num_replicas or (1 if debug else NUM_REPLICAS)

    # If running inside a Ray job, do the actual deployment
    if os.environ.get("_SERVER_INSIDE_RAY_JOB") == "1":
        model_path = os.environ["_SERVER_MODEL_PATH"]
        context_length = int(os.environ["_SERVER_CONTEXT_LENGTH"]) if os.environ.get("_SERVER_CONTEXT_LENGTH") else None
        tp = int(os.environ["_SERVER_TP"])
        gpus = int(os.environ["_SERVER_GPUS"])
        num_replicas = int(os.environ["_SERVER_NUM_REPLICAS"])
        _deploy(model_path, tp, gpus, num_replicas, context_length)
        return

    if args.clean_vram:
        kill_existing_sglang()

    model_path = resolve_model_path(debug)
    context_length = args.context_length
    ray_address = resolve_ray_address(args.ray_address)

    print(f"Model:    {model_path}")
    print(f"Replicas: {num_replicas}  GPUs/replica: {gpus}  TP: {tp}")
    print(f"Ray:      {ray_address}")
    print(f"Submitting as Ray job...")

    # Self-submit as a Ray job so the driver runs on the head node
    script_path = Path(__file__).resolve()
    env_vars = {
        "_SERVER_INSIDE_RAY_JOB": "1",
        "_SERVER_MODEL_PATH": model_path,
        "_SERVER_TP": str(tp),
        "_SERVER_GPUS": str(gpus),
        "_SERVER_NUM_REPLICAS": str(num_replicas),
    }
    if context_length is not None:
        env_vars["_SERVER_CONTEXT_LENGTH"] = str(context_length)

    runtime_env = json.dumps({
        "env_vars": env_vars,
        "excludes": [
            "teacher-qwen-35/.venv/**",
            "teacher-qwen-35/.cache/**",
            "logs/**",
            "__pycache__/**",
            "**/__pycache__/**",
            "*.pyc",
        ],
    })

    cmd = [
        "ray", "job", "submit",
        f"--address={ray_address}",
        "--working-dir", str(script_path.parent),
        f"--runtime-env-json={runtime_env}",
        "--", "python", script_path.name,
    ]
    print(f"$ {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
