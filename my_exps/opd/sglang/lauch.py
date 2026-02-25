"""
Ray Serve version of launching serve_teacher.sh on each node.

Run with:

bash my_exps/opd/sglang/submit_teacher_serve.sh


Client
   ↓
http://head-node:8000/teacher
   ↓
Gateway (Serve)
   ↓
TeacherWorker replicas (load balanced)
   ↓
Local subprocess at 127.0.0.1:13142  
"""

import os
import subprocess
from pathlib import Path

import ray
from ray import serve
import httpx
from starlette.responses import Response

# -----------------------------------------------------------------------------
# Cluster Config
# -----------------------------------------------------------------------------

ray.init(address="auto")

GPUS_PER_REPLICA = int(os.environ.get("TEACHER_GPUS_PER_ACTOR", "8"))
NUM_REPLICAS = int(os.environ.get("TEACHER_NUM_ACTORS", "16"))
ROUTE_PREFIX = "/teacher"
LOCAL_PORT = int(os.environ.get("TEACHER_LOCAL_PORT", "13142"))
LOCAL_BASE = f"http://127.0.0.1:{LOCAL_PORT}"

SCRIPT_DIR = Path(__file__).resolve().parent
SERVE_SCRIPT = (SCRIPT_DIR.parent / "serve_teacher.sh").resolve()

if not SERVE_SCRIPT.is_file():
    raise FileNotFoundError(f"serve_teacher.sh not found: {SERVE_SCRIPT}")

# -----------------------------------------------------------------------------
# Start Serve (detached so it survives driver exit)
# -----------------------------------------------------------------------------

serve.start(detached=True)

# -----------------------------------------------------------------------------
# Worker Deployment
# Each replica:
#   - Reserves GPUs
#   - Starts serve_teacher.sh
#   - Forwards requests to local :13142
# -----------------------------------------------------------------------------

@serve.deployment(
    name="TeacherWorker",
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_gpus": GPUS_PER_REPLICA},
)
class TeacherWorker:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=None)
        self.proc: subprocess.Popen | None = None
        self._start_teacher()

    def _start_teacher(self) -> None:
        # IMPORTANT: serve_teacher.sh must bind to 0.0.0.0:${LOCAL_PORT}
        self.proc = subprocess.Popen(["bash", str(SERVE_SCRIPT)])
        print(f"Started teacher subprocess PID={self.proc.pid}")

    def _ensure_teacher_alive(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            print("Teacher subprocess is not alive. Restarting...")
            self._start_teacher()

    async def __call__(self, request):
        self._ensure_teacher_alive()

        # Forward method/path/query so /teacher/v1/* maps to local /v1/*.
        incoming_path = request.url.path or "/"
        if incoming_path.startswith(ROUTE_PREFIX):
            upstream_path = incoming_path[len(ROUTE_PREFIX) :] or "/"
        else:
            upstream_path = incoming_path
        if not upstream_path.startswith("/"):
            upstream_path = f"/{upstream_path}"
        upstream_url = f"{LOCAL_BASE}{upstream_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        # Preserve auth/content headers for OpenAI-compatible API calls.
        hop_by_hop = {"host", "content-length", "connection", "accept-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop}
        payload = await request.body()
        response = await self.client.request(
            request.method,
            upstream_url,
            headers=headers,
            content=payload,
        )
        response_headers = {}
        content_type = response.headers.get("content-type")
        if content_type:
            response_headers["content-type"] = content_type
        return Response(content=response.content, status_code=response.status_code, headers=response_headers)


# -----------------------------------------------------------------------------
# Deploy Everything
# -----------------------------------------------------------------------------

app = TeacherWorker.bind()
serve.run(app, name="teacher-gateway", route_prefix=ROUTE_PREFIX)

print("Ray Serve deployment complete.")
print("Gateway available at: http://<head-node>:8000/teacher")
