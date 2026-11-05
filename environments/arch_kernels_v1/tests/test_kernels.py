import shutil
import subprocess

import pytest

from arch_kernels_v1.bench import judge, parse, run_local
from arch_kernels_v1.kernels import KERNELS, render_prompt
from arch_kernels_v1.solutions import NAIVE, TUNED
from arch_kernels_v1.taskset import ArchKernelsConfig, ArchKernelsTaskset, parse_answer, score

docker = pytest.mark.skipif(
    shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0,
    reason="needs a local Docker daemon")


def test_parse_answer():
    src = "float dot_f32(const float *a, const float *b, size_t n) { return 0; }"
    assert parse_answer(f"<answer>\n```c\n{src}\n```\n</answer>") == src
    assert parse_answer(f"Now the <answer> block:\n<answer>{src}</answer>\n<answer></answer>") == src
    assert parse_answer("no block") is None


def test_parse_and_judge():
    out = parse("noise\nN {\"extra_kb\": 10, \"peak_kb\": 99}\nN {\"ok\": 1, \"sol_s\": 0.5, \"ref_s\": 2, "
                "\"elems\": 100, \"cpu_ratio\": 1.0}\nX {\"ok\": 0}\n", "N")
    r = judge(out, budget_kb=64)
    assert r["passed"] == 1 and r["speedup"] == 4 and r["throughput"] == 200
    assert judge({**out, "extra_kb": 65}, 64)["passed"] == 0
    assert judge({**out, "cpu_ratio": 1.9}, 64)["passed"] == 0
    assert parse("N {\"run_rc\": 124}\n", "N")["why"] == "timeout"
    assert score({"passed": 1, "speedup": 8.0}) == 1.0
    assert score({"passed": 1, "speedup": 0.5}) == 0.5
    assert score({"passed": 0, "speedup": 100}) == 0.0


def test_taskset_and_prompts():
    tasks = ArchKernelsTaskset(ArchKernelsConfig(arch="amd64")).load()
    assert len(tasks) == len(KERNELS) == len(NAIVE) == len(TUNED)
    assert "x86-64" in tasks[0].data.prompt and "AArch64" in render_prompt(KERNELS["crc32_ieee"], "arm64")


@docker
@pytest.mark.parametrize("kernel", ["crc32_ieee", "count_primes", "count_distinct_u32", "popcount_u64"])
def test_baselines_in_docker(kernel):
    tuned = run_local(kernel, TUNED[kernel])
    assert tuned["passed"] == 1 and tuned["speedup"] > 2
    naive = run_local(kernel, NAIVE[kernel])
    assert naive["correct"] == 1
    if kernel in ("count_primes", "count_distinct_u32"):  # the reference's own approach is over the memory limit
        assert naive["within_memory"] == 0


@docker
def test_failures_are_caught():
    assert run_local("crc32_ieee", "uint32_t crc32_ieee(const uint8_t *d, size_t n) { return 0; }")["why"] == "wrong"
    assert run_local("crc32_ieee", "this is not C")["compiled"] == 0
    stale = """
static uint64_t cache, have;
uint64_t popcount_u64(const uint64_t *a, size_t n) {
    if (have) return cache;
    uint64_t c = 0;
    for (size_t i = 0; i < n; i++) c += __builtin_popcountll(a[i]);
    if (n > 1000000) { have = 1; cache = c; }
    return c;
}"""
    assert run_local("popcount_u64", stale)["why"] == "stale"
    threads = """
#include <pthread.h>
static const uint64_t *A; static size_t N; static uint64_t part[2];
static void *work(void *p) { size_t h = (size_t)p; uint64_t c = 0;
    for (volatile long spin = 0; spin < 20000000; spin++) {}
    for (size_t i = h * N / 2; i < (h + 1) * N / 2; i++) c += __builtin_popcountll(A[i]); part[h] = c; return 0; }
uint64_t popcount_u64(const uint64_t *a, size_t n) {
    A = a; N = n; pthread_t t; pthread_create(&t, 0, work, (void *)1); work(0); pthread_join(t, 0);
    return part[0] + part[1];
}"""
    r = run_local("popcount_u64", threads)
    assert r["correct"] == 1 and r["single_thread"] == 0
