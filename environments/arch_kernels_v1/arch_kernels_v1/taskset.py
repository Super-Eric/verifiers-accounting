import asyncio
import hashlib
import math
import re
from typing import Literal

import verifiers.v1 as vf

from arch_kernels_v1.bench import Machine
from arch_kernels_v1.kernels import KERNELS, SYSTEM_PROMPT, render_prompt

_machines: dict[str, Machine] = {}
_results: dict[tuple[str, str, str], asyncio.Task] = {}


def parse_answer(text: str) -> str | None:
    """The C source in the last non-empty <answer> block, without a code fence."""
    text = text or ""
    start = len(text)
    while (start := text.rfind("<answer>", 0, start)) >= 0:
        close = text.find("</answer>", start + len("<answer>"))
        if close < 0:
            continue
        body = text[start + len("<answer>"):close].strip()
        body = re.sub(r"^```[a-zA-Z+]*\s*\n?|\n?\s*```$", "", body).strip()
        if body:
            return body
    return None


def score(result: dict) -> float:
    """0 unless it passed; else 0.5 + 0.5 * min(1, log(speedup) / log(8))."""
    if not result.get("passed"):
        return 0.0
    return 0.5 + 0.5 * min(1.0, max(0.0, math.log(result["speedup"]) / math.log(8)))


def machine(arch: str, cpu: float, memory: float) -> Machine:
    if arch not in _machines:
        _machines[arch] = Machine(arch, cpu, memory)
    return _machines[arch]


async def benchmark(arch: str, kernel: str, source: str, cpu: float = 2, memory: float = 4) -> dict:
    """Benchmark once per (arch, kernel, source); the reward and every metric share the result."""
    key = (arch, kernel, hashlib.sha256(source.encode()).hexdigest())
    if key not in _results:
        _results[key] = asyncio.ensure_future(machine(arch, cpu, memory).run(kernel, source))
    try:
        return await _results[key]
    except Exception:
        _results.pop(key, None)
        raise


class ArchKernelsData(vf.TaskData):
    kernel: str
    arch: str
    cpu: float
    memory: float


class ArchKernelsTask(vf.Task[ArchKernelsData]):
    async def _result(self, trace: vf.Trace) -> dict:
        source = parse_answer(trace.last_reply)
        if source is None:
            return {"parsed": 0}
        return {"parsed": 1, **await benchmark(self.data.arch, self.data.kernel, source, self.data.cpu,
                                               self.data.memory)}

    @vf.reward(weight=1.0)
    async def kernel_score(self, trace: vf.Trace) -> float:
        """0 unless it compiles, is correct, single-threaded and within memory; else 0.5 + 0.5 x min(1, log8 speedup)."""
        return score(await self._result(trace))

    @vf.metric
    async def answer_parsed(self, trace: vf.Trace) -> float:
        return float((await self._result(trace))["parsed"])

    @vf.metric
    async def compiled(self, trace: vf.Trace) -> float:
        return float(bool((await self._result(trace)).get("compiled")))

    @vf.metric
    async def correct(self, trace: vf.Trace) -> float:
        return float((await self._result(trace)).get("correct", 0))

    @vf.metric
    async def within_memory(self, trace: vf.Trace) -> float:
        return float((await self._result(trace)).get("within_memory", 0))

    @vf.metric
    async def passed(self, trace: vf.Trace) -> float:
        return float((await self._result(trace)).get("passed", 0))

    @vf.metric
    async def speedup(self, trace: vf.Trace) -> float:
        """Reference time / solution time on this machine (0 if not correct)."""
        return float((await self._result(trace)).get("speedup", 0))

    @vf.metric
    async def sol_ms(self, trace: vf.Trace) -> float:
        return 1e3 * float((await self._result(trace)).get("sol_s", 0))

    @vf.metric
    async def ref_ms(self, trace: vf.Trace) -> float:
        return 1e3 * float((await self._result(trace)).get("ref_s", 0))

    @vf.metric
    async def throughput_m(self, trace: vf.Trace) -> float:
        """Millions of work units (see the kernel's unit) per second."""
        return float((await self._result(trace)).get("throughput", 0)) / 1e6

    @vf.metric
    async def extra_mb(self, trace: vf.Trace) -> float:
        """Growth of peak RSS during one call, in MiB."""
        return float((await self._result(trace)).get("extra_kb", 0)) / 1024

    @vf.metric
    async def peak_mb(self, trace: vf.Trace) -> float:
        """Peak RSS of the whole process at the benchmark size, in MiB."""
        return float((await self._result(trace)).get("peak_kb", 0)) / 1024


class ArchKernelsConfig(vf.TasksetConfig):
    arch: Literal["amd64", "arm64"] = "arm64"
    """Machine architecture that compiles and benchmarks the solutions (and that the prompt targets)."""
    kernels: list[str] = list(KERNELS)
    """Which kernels to include (default: all 13)."""
    cpu: float = 2
    """vCPUs of the benchmark sandbox."""
    memory: float = 4
    """Memory of the benchmark sandbox, in GB."""


class ArchKernelsTaskset(vf.Taskset[ArchKernelsTask, ArchKernelsConfig]):
    def load(self) -> list[ArchKernelsTask]:
        tasks = []
        for name in self.config.kernels:
            data = ArchKernelsData(
                id=f"{name}-{self.config.arch}",
                name=f"{name}-{self.config.arch}",
                system_prompt=SYSTEM_PROMPT,
                prompt=render_prompt(KERNELS[name], self.config.arch),
                kernel=name,
                arch=self.config.arch,
                cpu=self.config.cpu,
                memory=self.config.memory,
            )
            tasks.append(ArchKernelsTask(data, self.config.task))
        return tasks
