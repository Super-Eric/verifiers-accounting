"""Baseline solutions: NAIVE mirrors each reference (speedup about 1); TUNED is hand-optimized portable C with no
intrinsics, so the same source can be compared on both architectures."""

NAIVE = {
    "dot_f32": """
float dot_f32(const float *a, const float *b, size_t n) {
    float s = 0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}""",
    "prefix_sum_i64": """
void prefix_sum_i64(const int64_t *in, int64_t *out, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; i++) { s += in[i]; out[i] = s; }
}""",
    "byte_histogram": """
void byte_histogram(const uint8_t *data, size_t n, uint64_t counts[256]) {
    memset(counts, 0, 256 * sizeof counts[0]);
    for (size_t i = 0; i < n; i++) counts[data[i]]++;
}""",
    "count_primes": """
uint64_t count_primes(uint64_t n) {
    char *s = calloc(n + 1, 1);
    uint64_t c = 0;
    for (uint64_t i = 2; i <= n; i++)
        if (!s[i]) { c++; for (uint64_t j = i * i; j <= n; j += i) s[j] = 1; }
    free(s);
    return c;
}""",
    "matmul_f32": """
void matmul_f32(const float *A, const float *B, float *C, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            float s = 0;
            for (size_t k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}""",
    "sort_u32": """
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
void sort_u32(uint32_t *a, size_t n) { qsort(a, n, 4, cmp_u32); }""",
    "count_substr": """
uint64_t count_substr(const char *text, size_t n, const char *pat, size_t m) {
    uint64_t c = 0;
    if (m <= n)
        for (size_t i = 0; i + m <= n; i++) {
            size_t j = 0;
            while (j < m && text[i + j] == pat[j]) j++;
            c += j == m;
        }
    return c;
}""",
    "popcount_u64": """
uint64_t popcount_u64(const uint64_t *a, size_t n) {
    uint64_t c = 0;
    for (size_t i = 0; i < n; i++) for (uint64_t x = a[i]; x; x &= x - 1) c++;
    return c;
}""",
    "count_distinct_u32": """
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
size_t count_distinct_u32(const uint32_t *a, size_t n) {
    uint32_t *c = malloc(n * 4 + 4);
    memcpy(c, a, n * 4);
    qsort(c, n, 4, cmp_u32);
    size_t d = 0;
    for (size_t i = 0; i < n; i++) d += i == 0 || c[i] != c[i - 1];
    free(c);
    return d;
}""",
    "transpose_f32": """
void transpose_f32(const float *in, float *out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++) out[j * rows + i] = in[i * cols + j];
}""",
    "crc32_ieee": """
uint32_t crc32_ieee(const uint8_t *data, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= data[i];
        for (int b = 0; b < 8; b++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
    }
    return ~c;
}""",
    "levenshtein": """
size_t levenshtein(const char *s, size_t n, const char *t, size_t m) {
    size_t *row = malloc((m + 1) * sizeof *row);
    for (size_t j = 0; j <= m; j++) row[j] = j;
    for (size_t i = 1; i <= n; i++) {
        size_t prev = row[0];
        row[0] = i;
        for (size_t j = 1; j <= m; j++) {
            size_t cur = row[j], v = prev + (s[i - 1] != t[j - 1]);
            if (row[j] + 1 < v) v = row[j] + 1;
            if (row[j - 1] + 1 < v) v = row[j - 1] + 1;
            row[j] = v;
            prev = cur;
        }
    }
    size_t d = row[m];
    free(row);
    return d;
}""",
    "mandel_count": """
uint64_t mandel_count(int w, int h, int max_iter) {
    uint64_t c = 0;
    for (int py = 0; py < h; py++)
        for (int px = 0; px < w; px++) {
            double cx = -2.0 + 3.0 * (px + 0.5) / w, cy = -1.5 + 3.0 * (py + 0.5) / h, x = 0, y = 0;
            int i = 0;
            while (i < max_iter && x * x + y * y <= 4.0) { double xt = x * x - y * y + cx; y = 2 * x * y + cy; x = xt; i++; }
            c += i == max_iter;
        }
    return c;
}""",
}

TUNED = {
    "dot_f32": """
float dot_f32(const float *a, const float *b, size_t n) {
    float acc[32] = {0};
    size_t i = 0;
    for (; i + 32 <= n; i += 32)
        for (int j = 0; j < 32; j++) acc[j] += a[i + j] * b[i + j];
    float s = 0;
    for (int j = 0; j < 32; j++) s += acc[j];
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}""",
    "prefix_sum_i64": """
void prefix_sum_i64(const int64_t *in, int64_t *out, size_t n) {
    int64_t s = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int64_t a = s + in[i], b = a + in[i + 1], c = b + in[i + 2], d = c + in[i + 3];
        out[i] = a; out[i + 1] = b; out[i + 2] = c; out[i + 3] = d;
        s = d;
    }
    for (; i < n; i++) { s += in[i]; out[i] = s; }
}""",
    "byte_histogram": """
void byte_histogram(const uint8_t *data, size_t n, uint64_t counts[256]) {
    uint32_t h[4][256];
    memset(h, 0, sizeof h);
    memset(counts, 0, 256 * sizeof counts[0]);
    size_t i = 0;
    while (i < n) {
        size_t end = n - i > ((size_t)1 << 30) ? i + ((size_t)1 << 30) : n;
        for (; i + 4 <= end; i += 4) { h[0][data[i]]++; h[1][data[i + 1]]++; h[2][data[i + 2]]++; h[3][data[i + 3]]++; }
        for (; i < end; i++) h[0][data[i]]++;
        for (int b = 0; b < 256; b++) { counts[b] += (uint64_t)h[0][b] + h[1][b] + h[2][b] + h[3][b]; h[0][b] = h[1][b] = h[2][b] = h[3][b] = 0; }
    }
}""",
    "count_primes": """
uint64_t count_primes(uint64_t n) {
    if (n < 2) return 0;
    if (n < 3) return 1;
    uint64_t r = (uint64_t)sqrt((double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    uint8_t *small = calloc(r + 1, 1);
    uint32_t *pr = malloc(sizeof(uint32_t) * (r / 2 + 2));
    uint64_t *next = malloc(sizeof(uint64_t) * (r / 2 + 2));
    size_t np = 0;
    for (uint64_t i = 3; i <= r; i += 2)
        if (!small[i]) { pr[np] = (uint32_t)i; next[np++] = (i * i - 1) / 2; for (uint64_t j = i * i; j <= r; j += 2 * i) small[j] = 1; }
    enum { S = 1 << 15 };
    static uint8_t seg[S];
    uint64_t count = 1, kmax = (n - 1) / 2;   /* index k stands for the odd number 2k + 1 */
    for (uint64_t lo = 1; lo <= kmax; lo += S) {
        uint64_t hi = lo + S - 1 > kmax ? kmax : lo + S - 1;
        size_t len = hi - lo + 1;
        memset(seg, 0, len);
        for (size_t i = 0; i < np; i++) {
            uint64_t p = pr[i], k = next[i];
            for (; k <= hi; k += p) seg[k - lo] = 1;
            next[i] = k;
        }
        for (size_t j = 0; j < len; j++) count += !seg[j];
    }
    free(small); free(pr); free(next);
    return count;
}""",
    "matmul_f32": """
void matmul_f32(const float *A, const float *B, float *C, size_t n) {
    for (size_t i = 0; i < n * n; i++) C[i] = 0;
    const size_t KB = 128;
    for (size_t k0 = 0; k0 < n; k0 += KB) {
        size_t k1 = k0 + KB < n ? k0 + KB : n;
        for (size_t i = 0; i < n; i++) {
            float *c = C + i * n;
            for (size_t k = k0; k < k1; k++) {
                float a = A[i * n + k];
                const float *b = B + k * n;
                for (size_t j = 0; j < n; j++) c[j] += a * b[j];
            }
        }
    }
}""",
    "sort_u32": """
void sort_u32(uint32_t *a, size_t n) {
    if (n < 64) {
        for (size_t i = 1; i < n; i++) { uint32_t v = a[i]; size_t j = i; while (j && a[j - 1] > v) { a[j] = a[j - 1]; j--; } a[j] = v; }
        return;
    }
    uint32_t *tmp = malloc(n * 4), *src = a, *dst = tmp;
    for (int shift = 0; shift < 32; shift += 8) {
        size_t cnt[257] = {0};
        for (size_t i = 0; i < n; i++) cnt[((src[i] >> shift) & 255) + 1]++;
        for (int b = 0; b < 256; b++) cnt[b + 1] += cnt[b];
        for (size_t i = 0; i < n; i++) dst[cnt[(src[i] >> shift) & 255]++] = src[i];
        uint32_t *t = src; src = dst; dst = t;
    }
    free(tmp);
}""",
    "count_substr": """
uint64_t count_substr(const char *text, size_t n, const char *pat, size_t m) {
    if (m > n) return 0;
    uint64_t c = 0;
    const char p0 = pat[0], pl = pat[m - 1];
    for (size_t i = 0; i + m <= n; i++)
        c += text[i] == p0 && text[i + m - 1] == pl && memcmp(text + i, pat, m) == 0;
    return c;
}""",
    "popcount_u64": """
uint64_t popcount_u64(const uint64_t *a, size_t n) {
    uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        c0 += __builtin_popcountll(a[i]); c1 += __builtin_popcountll(a[i + 1]);
        c2 += __builtin_popcountll(a[i + 2]); c3 += __builtin_popcountll(a[i + 3]);
    }
    for (; i < n; i++) c0 += __builtin_popcountll(a[i]);
    return c0 + c1 + c2 + c3;
}""",
    "count_distinct_u32": """
size_t count_distinct_u32(const uint32_t *a, size_t n) {
    uint64_t *bm = calloc((size_t)1 << 20, 8);
    size_t d = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t v = a[i];
        uint64_t bit = 1ull << (v & 63), *w = bm + (v >> 6);
        d += !(*w & bit);
        *w |= bit;
    }
    free(bm);
    return d;
}""",
    "transpose_f32": """
void transpose_f32(const float *in, float *out, size_t rows, size_t cols) {
    const size_t T = 32;
    for (size_t i0 = 0; i0 < rows; i0 += T)
        for (size_t j0 = 0; j0 < cols; j0 += T) {
            size_t i1 = i0 + T < rows ? i0 + T : rows, j1 = j0 + T < cols ? j0 + T : cols;
            for (size_t i = i0; i < i1; i++)
                for (size_t j = j0; j < j1; j++) out[j * rows + i] = in[i * cols + j];
        }
}""",
    "crc32_ieee": """
static uint32_t crc_tab[8][256];
static int crc_ready;
uint32_t crc32_ieee(const uint8_t *data, size_t n) {
    if (!crc_ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int b = 0; b < 8; b++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
            crc_tab[0][i] = c;
        }
        for (int t = 1; t < 8; t++)
            for (int i = 0; i < 256; i++) crc_tab[t][i] = (crc_tab[t - 1][i] >> 8) ^ crc_tab[0][crc_tab[t - 1][i] & 255];
        crc_ready = 1;
    }
    uint32_t c = 0xFFFFFFFFu;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint32_t lo, hi;
        memcpy(&lo, data + i, 4); memcpy(&hi, data + i + 4, 4);
        lo ^= c;
        c = crc_tab[7][lo & 255] ^ crc_tab[6][(lo >> 8) & 255] ^ crc_tab[5][(lo >> 16) & 255] ^ crc_tab[4][lo >> 24] ^
            crc_tab[3][hi & 255] ^ crc_tab[2][(hi >> 8) & 255] ^ crc_tab[1][(hi >> 16) & 255] ^ crc_tab[0][hi >> 24];
    }
    for (; i < n; i++) c = (c >> 8) ^ crc_tab[0][(c ^ data[i]) & 255];
    return ~c;
}""",
    "levenshtein": """
size_t levenshtein(const char *s, size_t n, const char *t, size_t m) {
    if (n == 0) return m;
    if (m == 0) return n;
    uint32_t *row = malloc((m + 1) * sizeof *row);
    for (size_t j = 0; j <= m; j++) row[j] = (uint32_t)j;
    for (size_t i = 1; i <= n; i++) {
        uint32_t prev = row[0], left = (uint32_t)i;
        row[0] = left;
        const char si = s[i - 1];
        for (size_t j = 1; j <= m; j++) {
            uint32_t up = row[j], v = prev + (si != t[j - 1]);
            uint32_t w = (up < left ? up : left) + 1;
            v = v < w ? v : w;
            row[j] = left = v;
            prev = up;
        }
    }
    size_t d = row[m];
    free(row);
    return d;
}""",
    "mandel_count": """
uint64_t mandel_count(int w, int h, int max_iter) {
    uint64_t c = 0;
    for (int py = 0; py < h; py++) {
        double cy = -1.5 + 3.0 * (py + 0.5) / h, y2 = cy * cy;
        for (int px = 0; px < w; px++) {
            double cx = -2.0 + 3.0 * (px + 0.5) / w, q = (cx - 0.25) * (cx - 0.25) + y2;
            if (max_iter > 0 && (q * (q + (cx - 0.25)) <= 0.25 * y2 || (cx + 1) * (cx + 1) + y2 <= 0.0625)) { c++; continue; }
            double x = 0, y = 0;
            int i = 0;
            while (i < max_iter && x * x + y * y <= 4.0) { double xt = x * x - y * y + cx; y = 2 * x * y + cy; x = xt; i++; }
            c += i == max_iter;
        }
    }
    return c;
}""",
}
