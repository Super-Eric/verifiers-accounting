# arch-kernels-v1

Performance engineering on a real CPU, on the v1 `verifiers` stack (`import verifiers.v1 as vf`). The model writes a
fast C implementation of a small kernel for **x86-64 (amd64)** or **AArch64 (arm64)**. The answer is compiled and
benchmarked on a Prime sandbox of that architecture and scored on correctness, speedup over a reference and peak
memory.

The architecture is a config switch (`--env.taskset.arch amd64|arm64`), so one environment gives two otherwise
identical runs. The same machinery (`scripts/arch_compare.py`) builds and times any source on both machines. That
answers the question this env was built for: on equal vCPUs, which architecture gives more throughput and uses
less memory, for the reference code, for hand-tuned code and for code the models wrote.

## Kernels

| kernel | benchmark | what makes it fast | memory limit |
|---|---|---|---|
| `dot_f32` | 16M floats | many accumulators, SIMD (the reference is one dependent chain) | 4 MiB |
| `prefix_sum_i64` | 16M int64 | memory bandwidth | 4 MiB |
| `byte_histogram` | 64 MiB with long runs | several sub-histograms (store-to-load stalls) | 4 MiB |
| `count_primes` | n ≈ 5·10⁷ | segmented, odd-only sieve | 4 MiB (the reference's byte sieve needs 48) |
| `matmul_f32` | 384 × 384 | loop order, blocking, SIMD | 4 MiB |
| `sort_u32` | 4M uint32 | radix sort | 20 MiB |
| `count_substr` | 32 MiB text, 12-byte pattern | first/last-byte filter, SIMD compare | 4 MiB |
| `popcount_u64` | 32M words | popcount instructions | 4 MiB |
| `count_distinct_u32` | 4M values < 2²⁶ | an 8 MiB bitmap (sorting a copy needs 16) | 10 MiB |
| `transpose_f32` | 4096 × 4096 | tiling (a power-of-two stride thrashes the cache) | 4 MiB |
| `crc32_ieee` | 16 MiB | slicing tables, carry-less multiply, ARM's `crc32` instructions | 4 MiB |
| `levenshtein` | 6000 × 6000 | bit-parallel DP | 4 MiB |
| `mandel_count` | 1024², 1000 iterations | cardioid and bulb tests, SIMD | 4 MiB |

## Grading

The harness (`kernels.py`) puts the answer and the kernel's test code in one file and builds it with
`gcc -std=gnu11 -O3 -march=native` (gcc 14). It then runs:

1. **mem**: a fresh process sets up the benchmark inputs and records the growth of peak RSS (`ru_maxrss`) across one
   call. This counts malloc, touched static arrays and the stack.
2. **run**: every small test size (empty, tiny, odd lengths, unaligned pointers, the CRC check value), then the
   benchmark size. Solution and reference are timed on fresh inputs each repetition (median of 3–11). A final check
   runs on new inputs, so results cached across calls fail. CPU time over wall time > 1.25 means a second thread.

Results are compared with the same reference built the same way on the same machine, so speedups are
per-architecture. Every line the grader reads carries a random nonce, and runs are limited to 60 s.

| `kernel_score` | |
|---|---|
| 0 | no `<answer>`, does not compile, wrong, over the memory limit, more than one thread, or timeout |
| 0.5 + 0.5 · min(1, log speedup / log 8) | otherwise (a correct but slower answer scores 0.5; 8× or more scores 1) |

Metrics: `answer_parsed`, `compiled`, `correct`, `within_memory`, `passed`, `speedup`, `sol_ms`, `ref_ms`,
`throughput_m` (millions of the kernel's units per second), `extra_mb`, `peak_mb`.

Baselines (`solutions.py`): `NAIVE` is the reference code itself; `TUNED` is hand-optimized portable C with no
intrinsics, so the same source can be compared on both architectures.

## Config

`--env.taskset.arch` (`arm64`), `--env.taskset.kernels` (all 13), `--env.taskset.cpu` (2 vCPUs),
`--env.taskset.memory` (4 GB). One sandbox per architecture is started on the first answer and reused. Benchmarks run
one at a time, so they never share the CPU. Sandboxes delete themselves after 20 idle minutes.

Needs `prime login` (benchmarks run on Prime sandboxes). The tests use a local Docker daemon when one is available.

## Run

```bash
uv pip install -e environments/arch_kernels_v1
uv run vf-eval arch-kernels-v1 -m <model> -n 13 -r 3 --env.taskset.arch amd64 \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --client.base-url <openai-compatible-url> --client.api-key-var <VAR>
python scripts/arch_compare.py compare.json outputs/<run-dir> ...   # every answer on both architectures
```

Results are in [`results/arch-kernels-v1/`](../../results/arch-kernels-v1/README.md).
