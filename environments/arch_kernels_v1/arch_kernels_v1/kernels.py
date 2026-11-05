"""C kernels, their straightforward reference implementations, and the benchmark harness.

Each kernel is a C fragment that defines, after the model's solution in the same translation unit:

    K_NTESTS                 number of small correctness tests
    k_setup(int t)           allocate and fill test t (t < 0: the benchmark size); touch every buffer
    k_mutate(void)           fresh inputs of the same size, outputs poisoned (untimed)
    k_solve(void), k_ref(void)
    int k_check(void)        does the solution's output match the reference's?
    k_elems                  work units of the current size (for throughput)

The harness runs in two modes. `mem` starts a fresh process, sets up the benchmark size and records the growth in
peak RSS across one call of the solution. `run` checks every test size, then times the solution and the reference
alternately on fresh inputs, then checks once more on new inputs so that results cached across calls fail.
"""

from dataclasses import dataclass

COMPILE = "gcc -std=gnu11 -O3 -march=native -o k main.c -lm"


@dataclass(frozen=True)
class Kernel:
    name: str
    signature: str
    spec: str
    bench: str
    """The benchmark size, as told to the model."""
    unit: str
    """What k_elems counts (for throughput)."""
    budget_kb: int
    """Allowed growth of peak RSS during one call."""
    c: str


HARNESS = r"""
#define _GNU_SOURCE
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#line 1 "solution.c"
@@SOLUTION@@
#line 1 "harness.c"
static uint64_t k_s = 0x9E3779B97F4A7C15ull;
static uint64_t k_rnd(void) { k_s ^= k_s << 13; k_s ^= k_s >> 7; k_s ^= k_s << 17; return k_s; }
static double k_unif(void) { return (k_rnd() >> 11) * (1.0 / 9007199254740992.0); }
static double k_now(int clk) { struct timespec t; clock_gettime(clk, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static long k_rss_kb(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss; }
static double k_elems;
@@KERNEL@@
static int k_cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double k_cpu_ratio = 0;
static double k_bench(void (*f)(void)) {
    double t[11], wall = 0, cpu = 0;
    int r = 0;
    while (r < 11 && (r < 3 || wall < 0.4)) {
        k_mutate();
        void (*volatile call)(void) = f;   /* not inlined: the work cannot move past the clock reads */
        double c0 = k_now(CLOCK_PROCESS_CPUTIME_ID), w0 = k_now(CLOCK_MONOTONIC);
        __asm__ volatile("" ::: "memory");
        call();
        __asm__ volatile("" ::: "memory");
        double w1 = k_now(CLOCK_MONOTONIC), c1 = k_now(CLOCK_PROCESS_CPUTIME_ID);
        t[r++] = w1 - w0; wall += w1 - w0; cpu += c1 - c0;
    }
    if (wall > 0 && cpu / wall > k_cpu_ratio) k_cpu_ratio = cpu / wall;
    qsort(t, r, sizeof t[0], k_cmpd);
    return t[r / 2];
}
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const char *nonce = argv[2];
    if (!strcmp(argv[1], "mem")) {
        k_setup(-1);
        long r0 = k_rss_kb();
        k_solve();
        long r1 = k_rss_kb();
        printf("\n%s {\"extra_kb\": %ld, \"peak_kb\": %ld}\n", nonce, r1 - r0, r1);
        return 0;
    }
    for (int t = 0; t < K_NTESTS; t++) {
        k_s = 0x2545F4914F6CDD1Dull + 7919ull * t;
        k_setup(t);
        k_solve();
        k_ref();
        if (!k_check()) { printf("\n%s {\"ok\": 0, \"why\": \"wrong\", \"test\": %d}\n", nonce, t); return 0; }
    }
    k_s = 0x9E3779B97F4A7C15ull;
    k_setup(-1);
    k_solve();
    k_ref();
    if (!k_check()) { printf("\n%s {\"ok\": 0, \"why\": \"wrong\", \"test\": -1}\n", nonce); return 0; }
    double sol = k_bench(k_solve), sol_ratio = k_cpu_ratio;
    double ref = k_bench(k_ref);
    k_mutate();
    k_solve();
    k_ref();
    if (!k_check()) { printf("\n%s {\"ok\": 0, \"why\": \"stale\", \"test\": -1}\n", nonce); return 0; }
    printf("\n%s {\"ok\": 1, \"sol_s\": %.9g, \"ref_s\": %.9g, \"elems\": %.17g, \"cpu_ratio\": %.4f}\n",
           nonce, sol, ref, k_elems, sol_ratio);
    return 0;
}
"""


def program(kernel: Kernel, solution: str) -> str:
    return HARNESS.replace("@@SOLUTION@@", solution).replace("@@KERNEL@@", kernel.c)


_DOT = r"""
#define K_NTESTS 7
static const size_t k_sz[] = {0, 1, 3, 8, 17, 1000, 65537};
static float *k_a, *k_b;
static volatile float k_got, k_want;
static size_t k_n;
static void k_mutate(void) {
    for (size_t i = 0; i < k_n; i++) { k_a[i] = (float)(k_unif() * 2 - 1); k_b[i] = (float)(k_unif() * 2 - 1); }
    k_got = NAN;
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)16 << 20 : k_sz[t];
    k_a = malloc(k_n * 4 + 64); k_b = malloc(k_n * 4 + 64);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { k_got = dot_f32(k_a, k_b, k_n); }
static void k_ref(void) { float s = 0; for (size_t i = 0; i < k_n; i++) s += k_a[i] * k_b[i]; k_want = s; }
static int k_check(void) {
    double e = 0, m = 0;
    for (size_t i = 0; i < k_n; i++) { double p = (double)k_a[i] * k_b[i]; e += p; m += fabs(p); }
    return isfinite(k_got) && fabs(k_got - e) <= 1e-6 * m + 1e-3;
}
"""

_PREFIX = r"""
#define K_NTESTS 6
static const size_t k_sz[] = {0, 1, 2, 5, 1000, 100003};
static int64_t *k_in, *k_out, *k_rout;
static size_t k_n;
static void k_mutate(void) {
    for (size_t i = 0; i < k_n; i++) k_in[i] = (int64_t)(k_rnd() % 2000001) - 1000000;
    memset(k_out, 0xAB, k_n * 8);
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)16 << 20 : k_sz[t];
    k_in = malloc(k_n * 8 + 64); k_out = malloc(k_n * 8 + 64); k_rout = malloc(k_n * 8 + 64);
    memset(k_rout, 0, k_n * 8);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { prefix_sum_i64(k_in, k_out, k_n); }
static void k_ref(void) { int64_t s = 0; for (size_t i = 0; i < k_n; i++) { s += k_in[i]; k_rout[i] = s; } }
static int k_check(void) { return memcmp(k_out, k_rout, k_n * 8) == 0; }
"""

_HIST = r"""
#define K_NTESTS 7
static const size_t k_sz[] = {0, 1, 2, 255, 256, 1000, 4099};
static uint8_t *k_d;
static uint64_t k_c[256], k_rc[256];
static size_t k_n;
static void k_mutate(void) {
    size_t i = 0;
    while (i < k_n) {
        uint64_t r = k_rnd();
        size_t run = (r >> 8) & 1 ? 1 + ((r >> 9) & 63) : 1;
        for (size_t j = 0; j < run && i < k_n; j++) k_d[i++] = (uint8_t)r;
    }
    memset(k_c, 0xAB, sizeof k_c);
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)64 << 20 : k_sz[t];
    k_d = malloc(k_n + 64);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { byte_histogram(k_d, k_n, k_c); }
static void k_ref(void) { memset(k_rc, 0, sizeof k_rc); for (size_t i = 0; i < k_n; i++) k_rc[k_d[i]]++; }
static int k_check(void) { return memcmp(k_c, k_rc, sizeof k_c) == 0; }
"""

_PRIMES = r"""
#define K_NTESTS 10
static const uint64_t k_sz[] = {0, 1, 2, 3, 4, 10, 100, 1000, 7919, 1000003};
static uint64_t k_n, k_base;
static volatile uint64_t k_got, k_want;
static void k_mutate(void) { k_n = k_base > 1000 ? k_base - k_rnd() % 1000 : k_base; k_got = ~0ull; }
static void k_setup(int t) {
    k_base = t < 0 ? 50000000ull : k_sz[t];
    k_mutate();
    k_elems = (double)k_base;
}
static void k_solve(void) { k_got = count_primes(k_n); }
static void k_ref(void) {
    char *s = calloc(k_n + 1, 1);
    uint64_t c = 0;
    for (uint64_t i = 2; i <= k_n; i++)
        if (!s[i]) { c++; for (uint64_t j = i * i; j <= k_n; j += i) s[j] = 1; }
    free(s);
    k_want = c;
}
static int k_check(void) { return k_got == k_want; }
"""

_MATMUL = r"""
#define K_NTESTS 8
static const size_t k_sz[] = {0, 1, 2, 3, 7, 16, 33, 70};
static float *k_A, *k_B, *k_C, *k_R;
static size_t k_n;
static void k_mutate(void) {
    for (size_t i = 0; i < k_n * k_n; i++) { k_A[i] = (float)(k_unif() * 2 - 1); k_B[i] = (float)(k_unif() * 2 - 1); }
    memset(k_C, 0xFF, k_n * k_n * 4);
}
static void k_setup(int t) {
    k_n = t < 0 ? 384 : k_sz[t];
    size_t b = k_n * k_n * 4 + 64;
    k_A = malloc(b); k_B = malloc(b); k_C = malloc(b); k_R = malloc(b);
    memset(k_R, 0, b);
    k_mutate();
    k_elems = (double)k_n * k_n * k_n;
}
static void k_solve(void) { matmul_f32(k_A, k_B, k_C, k_n); }
static void k_ref(void) {
    for (size_t i = 0; i < k_n; i++)
        for (size_t j = 0; j < k_n; j++) {
            float s = 0;
            for (size_t k = 0; k < k_n; k++) s += k_A[i * k_n + k] * k_B[k * k_n + j];
            k_R[i * k_n + j] = s;
        }
}
static int k_check(void) {
    for (size_t i = 0; i < k_n; i++)
        for (size_t j = 0; j < k_n; j++) {
            double e = 0, m = 0;
            for (size_t k = 0; k < k_n; k++) { double p = (double)k_A[i * k_n + k] * k_B[k * k_n + j]; e += p; m += fabs(p); }
            float g = k_C[i * k_n + j];
            if (!isfinite(g) || fabs(g - e) > 1e-4 * m + 1e-5) return 0;
        }
    return 1;
}
"""

_SORT = r"""
#define K_NTESTS 7
static const size_t k_sz[] = {0, 1, 2, 3, 10, 1000, 65537};
static uint32_t *k_in, *k_out, *k_rout;
static size_t k_n;
static int k_cmpu(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }
static void k_mutate(void) {
    for (size_t i = 0; i < k_n; i++) k_in[i] = (k_rnd() & 3) ? (uint32_t)k_rnd() : (uint32_t)(k_rnd() % 1000);
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)4 << 20 : k_sz[t];
    k_in = malloc(k_n * 4 + 64); k_out = malloc(k_n * 4 + 64); k_rout = malloc(k_n * 4 + 64);
    memset(k_out, 0, k_n * 4); memset(k_rout, 0, k_n * 4);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { memcpy(k_out, k_in, k_n * 4); sort_u32(k_out, k_n); }
static void k_ref(void) { memcpy(k_rout, k_in, k_n * 4); qsort(k_rout, k_n, 4, k_cmpu); }
static int k_check(void) { return memcmp(k_out, k_rout, k_n * 4) == 0; }
"""

_SUBSTR = r"""
#define K_NTESTS 9
static const size_t k_tn[] = {0, 1, 5, 5, 100, 100, 1000, 5000, 4099};
static const size_t k_tm[] = {1, 1, 5, 6, 1, 3, 7, 64, 2};
static const int k_ta[] = {4, 4, 1, 1, 4, 1, 4, 2, 2};
static char *k_t, k_p[65];
static size_t k_n, k_m;
static int k_alph;
static volatile uint64_t k_got, k_want;
static void k_mutate(void) {
    for (size_t i = 0; i < k_n; i++) k_t[i] = 'a' + (char)(k_rnd() % k_alph);
    if (k_m <= k_n)
        for (size_t c = 0; c < k_n / 4096; c++) memcpy(k_t + k_rnd() % (k_n - k_m + 1), k_p, k_m);
    k_got = ~0ull;
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)32 << 20 : k_tn[t];
    k_m = t < 0 ? 12 : k_tm[t];
    k_alph = t < 0 ? 4 : k_ta[t];
    k_t = calloc(k_n + 64, 1);
    memset(k_p, 0, sizeof k_p);
    for (size_t j = 0; j < k_m; j++) k_p[j] = 'a' + (char)(k_rnd() % k_alph);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { k_got = count_substr(k_t, k_n, k_p, k_m); }
static void k_ref(void) {
    uint64_t c = 0;
    if (k_m <= k_n)
        for (size_t i = 0; i + k_m <= k_n; i++) {
            size_t j = 0;
            while (j < k_m && k_t[i + j] == k_p[j]) j++;
            c += j == k_m;
        }
    k_want = c;
}
static int k_check(void) { return k_got == k_want; }
"""

_POPCOUNT = r"""
#define K_NTESTS 6
static const size_t k_sz[] = {0, 1, 2, 3, 100, 1001};
static uint64_t *k_a;
static volatile uint64_t k_got, k_want;
static size_t k_n;
static void k_mutate(void) { for (size_t i = 0; i < k_n; i++) k_a[i] = k_rnd() & k_rnd() | (k_rnd() & 0xFF00FF); k_got = ~0ull; }
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)32 << 20 : k_sz[t];
    k_a = malloc(k_n * 8 + 64);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { k_got = popcount_u64(k_a, k_n); }
static void k_ref(void) {
    uint64_t c = 0;
    for (size_t i = 0; i < k_n; i++) for (uint64_t x = k_a[i]; x; x &= x - 1) c++;
    k_want = c;
}
static int k_check(void) { return k_got == k_want; }
"""

_DISTINCT = r"""
#define K_NTESTS 6
static const size_t k_sz[] = {0, 1, 2, 5, 1000, 65537};
static uint32_t *k_a;
static size_t k_n;
static volatile size_t k_got, k_want;
static int k_cmpu(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }
static void k_mutate(void) {
    for (size_t i = 0; i < k_n; i++) k_a[i] = (k_rnd() & 1) ? (uint32_t)(k_rnd() % (1u << 26)) : (uint32_t)(k_rnd() % 100000);
    k_got = (size_t)-1;
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)4 << 20 : k_sz[t];
    k_a = malloc(k_n * 4 + 64);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { k_got = count_distinct_u32(k_a, k_n); }
static void k_ref(void) {
    uint32_t *c = malloc(k_n * 4 + 4);
    memcpy(c, k_a, k_n * 4);
    qsort(c, k_n, 4, k_cmpu);
    size_t d = 0;
    for (size_t i = 0; i < k_n; i++) d += i == 0 || c[i] != c[i - 1];
    free(c);
    k_want = d;
}
static int k_check(void) { return k_got == k_want; }
"""

_TRANSPOSE = r"""
#define K_NTESTS 9
static const size_t k_tr[] = {0, 0, 1, 1, 9, 3, 17, 64, 100};
static const size_t k_tc[] = {0, 5, 1, 9, 1, 5, 33, 64, 37};
static float *k_in, *k_out, *k_rout;
static size_t k_r, k_c;
static void k_mutate(void) {
    for (size_t i = 0; i < k_r * k_c; i++) k_in[i] = (float)(k_unif() * 2 - 1);
    memset(k_out, 0xFF, k_r * k_c * 4);
}
static void k_setup(int t) {
    k_r = t < 0 ? 4096 : k_tr[t];
    k_c = t < 0 ? 4096 : k_tc[t];
    size_t b = k_r * k_c * 4 + 64;
    k_in = malloc(b); k_out = malloc(b); k_rout = malloc(b);
    memset(k_rout, 0, b);
    k_mutate();
    k_elems = (double)k_r * k_c;
}
static void k_solve(void) { transpose_f32(k_in, k_out, k_r, k_c); }
static void k_ref(void) {
    for (size_t i = 0; i < k_r; i++)
        for (size_t j = 0; j < k_c; j++) k_rout[j * k_r + i] = k_in[i * k_c + j];
}
static int k_check(void) { return memcmp(k_out, k_rout, k_r * k_c * 4) == 0; }
"""

_CRC = r"""
#define K_NTESTS 13
static const size_t k_sz[] = {0, 1, 2, 3, 7, 8, 9, 63, 64, 65, 1000, 4097, 9};
static uint8_t *k_buf, *k_d;
static size_t k_n;
static int k_fixed;
static volatile uint32_t k_got, k_want;
static void k_mutate(void) {
    if (k_fixed) memcpy(k_d, "123456789", 9);
    else for (size_t i = 0; i < k_n; i++) k_d[i] = (uint8_t)k_rnd();
    k_got = 0;
}
static void k_setup(int t) {
    k_n = t < 0 ? (size_t)16 << 20 : k_sz[t];
    k_fixed = t == K_NTESTS - 1;
    k_buf = malloc(k_n + 64);
    k_d = k_buf + (t < 0 ? 0 : t % 3);
    k_mutate();
    k_elems = (double)k_n;
}
static void k_solve(void) { k_got = crc32_ieee(k_d, k_n); }
static void k_ref(void) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < k_n; i++) {
        c ^= k_d[i];
        for (int b = 0; b < 8; b++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
    }
    k_want = ~c;
}
static int k_check(void) { return k_got == k_want && (!k_fixed || k_got == 0xCBF43926u); }
"""

_LEV = r"""
#define K_NTESTS 9
static const size_t k_tn[] = {0, 0, 5, 1, 3, 64, 65, 200, 1000};
static const size_t k_tm[] = {0, 5, 0, 1, 7, 64, 63, 150, 1200};
static char *k_x, *k_y;
static size_t k_n, k_m;
static volatile size_t k_got, k_want;
static void k_mutate(void) {
    for (size_t i = 0; i < k_n; i++) k_x[i] = 'a' + (char)(k_rnd() % 4);
    size_t j = 0;
    for (size_t i = 0; i < k_n && j < k_m; i++) {
        uint64_t r = k_rnd() % 30;
        if (r == 0) continue;                                   /* delete */
        if (r == 1) { k_y[j++] = 'a' + (char)(k_rnd() % 4); i--; continue; }   /* insert */
        k_y[j++] = r == 2 ? 'a' + (char)(k_rnd() % 4) : k_x[i];   /* substitute or keep */
    }
    while (j < k_m) k_y[j++] = 'a' + (char)(k_rnd() % 4);
    k_x[k_n] = k_y[k_m] = 0;
    k_got = (size_t)-1;
}
static void k_setup(int t) {
    k_n = t < 0 ? 6000 : k_tn[t];
    k_m = t < 0 ? 6000 : k_tm[t];
    k_x = calloc(k_n + 64, 1); k_y = calloc(k_m + 64, 1);
    k_mutate();
    k_elems = (double)k_n * k_m;
}
static void k_solve(void) { k_got = levenshtein(k_x, k_n, k_y, k_m); }
static void k_ref(void) {
    size_t *row = malloc((k_m + 1) * sizeof *row);
    for (size_t j = 0; j <= k_m; j++) row[j] = j;
    for (size_t i = 1; i <= k_n; i++) {
        size_t prev = row[0];
        row[0] = i;
        for (size_t j = 1; j <= k_m; j++) {
            size_t cur = row[j], v = prev + (k_x[i - 1] != k_y[j - 1]);
            if (row[j] + 1 < v) v = row[j] + 1;
            if (row[j - 1] + 1 < v) v = row[j - 1] + 1;
            row[j] = v;
            prev = cur;
        }
    }
    k_want = row[k_m];
    free(row);
}
static int k_check(void) { return k_got == k_want; }
"""

_MANDEL = r"""
#define K_NTESTS 5
static const int k_tw[] = {1, 2, 16, 64, 101}, k_th[] = {1, 3, 16, 48, 77}, k_ti[] = {10, 50, 100, 256, 500};
static int k_w, k_h, k_it, k_bw, k_bh, k_bi;
static volatile uint64_t k_got, k_want;
static void k_mutate(void) {
    k_w = k_bw > 200 ? k_bw - (int)(k_rnd() % 16) : k_bw;
    k_h = k_bh > 200 ? k_bh - (int)(k_rnd() % 16) : k_bh;
    k_it = k_bi > 200 ? k_bi - (int)(k_rnd() % 8) : k_bi;
    k_got = ~0ull;
}
static void k_setup(int t) {
    k_bw = t < 0 ? 1024 : k_tw[t];
    k_bh = t < 0 ? 1024 : k_th[t];
    k_bi = t < 0 ? 1000 : k_ti[t];
    k_mutate();
    k_elems = (double)k_bw * k_bh;
}
static void k_solve(void) { k_got = mandel_count(k_w, k_h, k_it); }
static void k_ref(void) {
    uint64_t c = 0;
    for (int py = 0; py < k_h; py++)
        for (int px = 0; px < k_w; px++) {
            double cx = -2.0 + 3.0 * (px + 0.5) / k_w, cy = -1.5 + 3.0 * (py + 0.5) / k_h, x = 0, y = 0;
            int i = 0;
            while (i < k_it && x * x + y * y <= 4.0) { double xt = x * x - y * y + cx; y = 2 * x * y + cy; x = xt; i++; }
            c += i == k_it;
        }
    k_want = c;
}
static int k_check(void) {
    double d = (double)k_got - (double)k_want;
    return k_got != ~0ull && fabs(d) <= 0.002 * k_w * k_h + 2;
}
"""

KERNELS: dict[str, Kernel] = {k.name: k for k in [
    Kernel("dot_f32", "float dot_f32(const float *a, const float *b, size_t n);",
           "Return the dot product sum(a[i] * b[i]). Values are in [-1, 1]. It must be within 1e-6 * sum|a[i]*b[i]| + "
           "1e-3 of the exact sum.", "n = 16M", "elements", 4096, _DOT),
    Kernel("prefix_sum_i64", "void prefix_sum_i64(const int64_t *in, int64_t *out, size_t n);",
           "Write the inclusive prefix sums: out[i] = in[0] + ... + in[i]. Values are in [-1e6, 1e6]; the buffers do "
           "not overlap.", "n = 16M", "elements", 4096, _PREFIX),
    Kernel("byte_histogram", "void byte_histogram(const uint8_t *data, size_t n, uint64_t counts[256]);",
           "Set counts[b] to the number of bytes equal to b, for all 256 values (counts arrives holding garbage). The "
           "data contains long runs of repeated bytes.", "n = 64 MiB", "bytes", 4096, _HIST),
    Kernel("count_primes", "uint64_t count_primes(uint64_t n);",
           "Return the number of primes <= n.", "n = about 50,000,000", "integers sieved", 4096, _PRIMES),
    Kernel("matmul_f32", "void matmul_f32(const float *A, const float *B, float *C, size_t n);",
           "C = A * B for row-major n x n matrices (C does not alias A or B). Values are in [-1, 1]; each entry must be "
           "within 1e-4 * sum_k |A[i][k] * B[k][j]| + 1e-5 of the exact value.", "n = 384", "multiply-adds", 4096,
           _MATMUL),
    Kernel("sort_u32", "void sort_u32(uint32_t *a, size_t n);",
           "Sort a ascending, in place. Values are arbitrary 32-bit, with many duplicates.", "n = 4M", "elements",
           20480, _SORT),
    Kernel("count_substr", "uint64_t count_substr(const char *text, size_t n, const char *pat, size_t m);",
           "Return the number of positions i (0 <= i <= n - m) with text[i..i+m) == pat[0..m), overlapping matches "
           "included; 0 if m > n. 1 <= m <= 64. Text and pattern use the letters a-d; neither is NUL-terminated, so "
           "do not read text[n] or beyond.", "n = 32 MiB, m = 12", "bytes", 4096, _SUBSTR),
    Kernel("popcount_u64", "uint64_t popcount_u64(const uint64_t *a, size_t n);",
           "Return the total number of set bits in a[0..n).", "n = 32M", "words", 4096, _POPCOUNT),
    Kernel("count_distinct_u32", "size_t count_distinct_u32(const uint32_t *a, size_t n);",
           "Return the number of distinct values in a[0..n). Every value is < 2^26. Do not modify a.", "n = 4M",
           "elements", 10240, _DISTINCT),
    Kernel("transpose_f32", "void transpose_f32(const float *in, float *out, size_t rows, size_t cols);",
           "in is rows x cols, row-major; write its transpose: out[j * rows + i] = in[i * cols + j]. The buffers do "
           "not overlap.", "rows = cols = 4096", "elements", 4096, _TRANSPOSE),
    Kernel("crc32_ieee", "uint32_t crc32_ieee(const uint8_t *data, size_t n);",
           "Return the standard CRC-32 (IEEE 802.3, as zlib's crc32(0, data, n)): reflected polynomial 0xEDB88320, "
           "initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF. crc32_ieee(\"123456789\", 9) == 0xCBF43926. data may be "
           "unaligned.", "n = 16 MiB", "bytes", 4096, _CRC),
    Kernel("levenshtein", "size_t levenshtein(const char *s, size_t n, const char *t, size_t m);",
           "Return the edit distance (insertions, deletions and substitutions all cost 1) between s[0..n) and t[0..m). "
           "Both use the letters a-d; t is s with about 10% random edits. n, m <= 20000.", "n = m = 6000",
           "DP cells", 4096, _LEV),
    Kernel("mandel_count", "uint64_t mandel_count(int w, int h, int max_iter);",
           "Pixel (px, py), 0 <= px < w, 0 <= py < h, is the point c = (-2 + 3 * (px + 0.5) / w) + "
           "(-1.5 + 3 * (py + 0.5) / h) i. Iterate z = z^2 + c from z = 0 up to max_iter times, stopping early once "
           "|z|^2 > 4. Return how many pixels never stop early. Counts within 0.2% of w * h of the double-precision "
           "reference are accepted.", "w = h = 1024, max_iter = 1000", "pixels", 4096, _MANDEL),
]}

ARCH_LABEL = {"amd64": "x86-64 (amd64)", "arm64": "AArch64 (arm64)"}

SYSTEM_PROMPT = """\
You write fast, correct C. You cannot compile or run anything: reason about the target machine and the compiler.
Finish with exactly one block holding the complete C source:
<answer>
```c
...
```
</answer>"""


def render_prompt(kernel: Kernel, arch: str) -> str:
    return f"""Implement this function as fast as you can for {ARCH_LABEL[arch]} Linux:

    {kernel.signature}

{kernel.spec}

Build: one translation unit, `{COMPILE}` with gcc 14. The harness already includes math.h, stddef.h, stdint.h, \
stdio.h, stdlib.h, string.h, sys/resource.h and time.h; add any other headers you need, including SIMD intrinsics \
headers for this architecture.

Benchmark: {kernel.bench}, on a cloud VM. Your median time is compared with a straightforward reference \
implementation built the same way on the same machine.

Rules:
- Correct on every test, including empty and tiny sizes.
- Single thread. Inputs change between calls, so keep no results across calls.
- One call may grow the process's peak memory by at most {kernel.budget_kb // 1024} MiB beyond the inputs and \
outputs (malloc, static arrays and stack all count).
- Define {kernel.signature.split('(')[0].split()[-1].lstrip('*')} (and any helpers). Do not define main or names \
starting with k_, and do no I/O.

Score: 0 if it does not compile, is wrong, uses more than one thread, exceeds the memory limit or takes over 60 s. \
Otherwise 0.5 + 0.5 * min(1, log(speedup) / log(8)), so an 8x speedup over the reference scores 1."""
