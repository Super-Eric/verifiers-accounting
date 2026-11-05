"""Compile and benchmark a solution on a real amd64 or arm64 machine (a Prime sandbox), or in a local container."""

import asyncio
import json
import secrets
import subprocess
import time
import uuid

from arch_kernels_v1.kernels import COMPILE, KERNELS, program

IMAGE = "python:3.11-slim"
"""Debian 13; SETUP installs its gcc 14.2. (Prebuilt images reach both architectures faster than gcc:14.)"""
LOCAL_IMAGE = "gcc:14"
RUN_TIMEOUT = 60
SETUP = ("command -v gcc >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
         "apt-get install -y -qq gcc libc6-dev >/dev/null; }")
PROBE = ("uname -m; nproc; getconf PAGESIZE; gcc -dumpfullversion; "
         "lscpu | grep -E '^(Model name|Vendor ID|Thread\\(s\\) per core|L1d cache|L2 cache|L3 cache):' | sed 's/  */ /g'")


def script(nonce: str) -> str:
    """Shell steps run in the job directory; every line of output that matters starts with the nonce."""
    tag = f"echo; echo {nonce}"
    return (f"timeout 60 {COMPILE} 2> cc.err; rc=$?; "
            f"if [ $rc -ne 0 ]; then {tag} '{{\"compiled\": 0}}'; head -c 2000 cc.err; exit 0; fi; "
            f"timeout {RUN_TIMEOUT} ./k mem {nonce}; rc=$?; {tag} \"{{\\\"mem_rc\\\": $rc}}\"; "
            f"timeout {RUN_TIMEOUT} ./k run {nonce}; rc=$?; {tag} \"{{\\\"run_rc\\\": $rc}}\"")


def parse(stdout: str, nonce: str) -> dict:
    """Merge the nonce-tagged JSON lines into one result."""
    out = {"compiled": 1}
    for line in stdout.splitlines():
        if line.startswith(nonce + " "):
            try:
                out.update(json.loads(line[len(nonce) + 1:]))
            except ValueError:
                pass
    if out["compiled"] and "ok" not in out:
        out.update(ok=0, why="timeout" if out.get("run_rc") == 124 else "crash")
    return out


def judge(result: dict, budget_kb: int) -> dict:
    """Add verdicts: correct, within_memory, single_thread, passed, speedup."""
    r = dict(result)
    r["correct"] = int(bool(r.get("compiled")) and r.get("ok") == 1)
    r["within_memory"] = int("extra_kb" in r and r["extra_kb"] <= budget_kb)
    r["single_thread"] = int(r.get("cpu_ratio", 0) <= 1.25)
    r["passed"] = int(r["correct"] and r["within_memory"] and r["single_thread"])
    if r["correct"]:
        r["speedup"] = r["ref_s"] / max(r["sol_s"], 1e-9)
        r["throughput"] = r["elems"] / max(r["sol_s"], 1e-9)
        r["ref_throughput"] = r["elems"] / max(r["ref_s"], 1e-9)
    return r


class Machine:
    """One long-lived sandbox per architecture; benchmarks run one at a time so they do not share the CPU."""

    def __init__(self, arch: str, cpu: float = 2, memory: float = 4, labels: list[str] | None = None):
        self.arch, self.cpu, self.memory = arch, cpu, memory
        self.labels = labels or ["arch-kernels-v1"]
        self.client = None
        self.sandbox_id: str | None = None
        self.info: str | None = None
        self._start = asyncio.Lock()
        self._run = asyncio.Lock()

    async def _exec(self, cmd: str, timeout: int = 300):
        for attempt in range(20):  # transient gateway errors ("being placed on a node") can last minutes
            try:
                return await self.client.execute_command(self.sandbox_id, cmd, timeout=timeout)
            except Exception:
                if attempt == 19:
                    raise
                await asyncio.sleep(min(30, 5 * (attempt + 1)))

    async def start(self) -> None:
        async with self._start:
            if self.sandbox_id:
                return
            from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

            self.client = AsyncSandboxClient()
            sb = await self.client.create(CreateSandboxRequest(
                name=f"arch-kernels-{self.arch}-{uuid.uuid4().hex[:6]}", docker_image=IMAGE, arch=self.arch,
                cpu_cores=self.cpu, memory_gb=self.memory, disk_size_gb=5, timeout_minutes=240,
                idle_timeout_minutes=20, labels=self.labels))
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                try:
                    status = (await self.client.get(sb.id)).status
                except Exception:
                    status = "UNKNOWN"
                if status == "RUNNING":
                    break
                if status in ("ERROR", "FAILED", "TERMINATED"):
                    raise RuntimeError(f"{self.arch} sandbox {sb.id} is {status}")
                await asyncio.sleep(10)
            self.sandbox_id = sb.id
            self.info = (await self._exec(f"{SETUP}; {PROBE}", timeout=600)).stdout.strip()

    async def run(self, kernel: str, solution: str) -> dict:
        await self.start()
        nonce = secrets.token_hex(8)
        job = f"/tmp/j{nonce}"
        async with self._run:
            await self.client.upload_bytes(self.sandbox_id, f"{job}/main.c", program(KERNELS[kernel], solution).encode(),
                                           "main.c")
            res = await self._exec(f"cd {job} && ( {script(nonce)} ); cd / && rm -rf {job}",
                                   timeout=3 * RUN_TIMEOUT + 60)
        out = parse(res.stdout, nonce)
        if not out.get("compiled"):
            out["compile_error"] = res.stdout.split(nonce, 1)[-1][-2000:]
        return judge(out, KERNELS[kernel].budget_kb)

    async def close(self) -> None:
        if self.sandbox_id:
            try:
                await self.client.delete(self.sandbox_id)
            finally:
                self.sandbox_id = None


def run_local(kernel: str, solution: str, image: str = LOCAL_IMAGE) -> dict:
    """The same steps in a local Docker container (for tests; native architecture of this machine)."""
    nonce = secrets.token_hex(8)
    src = program(KERNELS[kernel], solution)
    cmd = f"mkdir -p /w && cd /w && cat > main.c && ( {script(nonce)} )"
    res = subprocess.run(["docker", "run", "--rm", "-i", image, "bash", "-c", cmd], input=src, capture_output=True,
                         text=True, timeout=600)
    out = parse(res.stdout, nonce)
    if not out.get("compiled"):
        out["compile_error"] = res.stdout.split(nonce, 1)[-1][-2000:]
    return judge(out, KERNELS[kernel].budget_kb)

