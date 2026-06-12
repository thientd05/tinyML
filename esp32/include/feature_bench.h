// feature_bench.h — times the RF/SVM feature-extraction front-end on the ESP32.
//
// RF/SVM are fed the 21-D feature vector; on PC that vector is precomputed, so their raw
// inference latency (~tens of us) HIDES the cost of producing the features (db4 wavelet +
// FFT + time/freq stats). CNN/LSTM eat the raw beat, so their feature cost is 0. To make
// the cross-family "detection time" apples-to-apples we measure this front-end once here
// and add its us/beat to every RF/SVM point in src/analyze.py (--feature-us).
//
// This mirrors the heavy ops of src/utils.extract_features for an honest timing (exact
// boundary handling / FFT length differ slightly from numpy — FFT is zero-padded to a
// power of two as a real deployment would — but the compute is representative). Outputs
// are discarded; only wall time matters.
#pragma once
#include <Arduino.h>
#include <math.h>

#ifndef RAW_LEN
#define RAW_LEN 200
#endif
#define FEAT_FFT_N 256          // next pow2 >= RAW_LEN (zero-padded), for radix-2 FFT

// db4 (Daubechies-4, 8 taps) decomposition low-pass; high-pass derived via QMF.
static const float DB4_LO[8] = {
  -0.0105974018f, 0.0328830117f, 0.0308413818f, -0.1870348117f,
  -0.0279837694f, 0.6308807679f, 0.7148465706f, 0.2303778133f};

// one DWT level (periodic convolution + downsample by 2)
static inline void dwt_level(const float *x, int n, float *approx, float *detail) {
  int half = n / 2;
  for (int i = 0; i < half; i++) {
    float a = 0.0f, d = 0.0f;
    for (int k = 0; k < 8; k++) {
      int idx = (2 * i + k) % n;
      float hi = ((k & 1) ? -1.0f : 1.0f) * DB4_LO[7 - k];  // QMF high-pass
      a += DB4_LO[k] * x[idx];
      d += hi * x[idx];
    }
    approx[i] = a;
    detail[i] = d;
  }
}

// iterative in-place radix-2 FFT (n must be a power of two)
static inline void fft_radix2(float *re, float *im, int n) {
  for (int i = 1, j = 0; i < n; i++) {            // bit-reversal permutation
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / len;
    float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = a + len / 2;
        float xr = re[b] * cr - im[b] * ci;
        float xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr;        im[a] += xi;
        float ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

// run the heavy feature ops on one beat; result discarded (timing only)
static void feature_extract_one(const float *beat, int len) {
  volatile float sink = 0.0f;

  // time-domain: mean/std/min/max/skew/kurtosis/rms/p2p/energy/zero-crossings
  float mn = beat[0], mx = beat[0], sum = 0.0f, sq = 0.0f;
  int zc = 0;
  for (int i = 0; i < len; i++) {
    float v = beat[i];
    sum += v; sq += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    if (i && (beat[i - 1] * v) < 0.0f) zc++;
  }
  float mean = sum / len;
  float var = sq / len - mean * mean; if (var < 0) var = 0;
  float sd = sqrtf(var);
  float m3 = 0.0f, m4 = 0.0f;
  for (int i = 0; i < len; i++) { float d = beat[i] - mean; m3 += d * d * d; m4 += d * d * d * d; }
  float skew = (sd > 1e-8f) ? (m3 / len) / (sd * sd * sd) : 0.0f;
  float kurt = (var > 1e-12f) ? (m4 / len) / (var * var) : 0.0f;
  sink += mean + sd + mn + mx + skew + kurt + sqrtf(sq / len) + (mx - mn) + sq + zc;

  // db4 wavelet, 4 levels -> 5 sub-band energies
  static float buf[RAW_LEN], a[RAW_LEN], d[RAW_LEN];
  for (int i = 0; i < len; i++) buf[i] = beat[i];
  int n = len;
  for (int lvl = 0; lvl < 4; lvl++) {
    dwt_level(buf, n, a, d);
    int half = n / 2;
    float eD = 0.0f;
    for (int i = 0; i < half; i++) eD += d[i] * d[i];
    for (int i = 0; i < half; i++) buf[i] = a[i];
    n = half;
    sink += eD;
  }
  float eA = 0.0f;
  for (int i = 0; i < n; i++) eA += buf[i] * buf[i];
  sink += eA;

  // FFT (zero-pad to FEAT_FFT_N) -> dominant freq + spectral entropy
  static float re[FEAT_FFT_N], im[FEAT_FFT_N], pw[FEAT_FFT_N / 2];
  for (int i = 0; i < FEAT_FFT_N; i++) { re[i] = (i < len) ? beat[i] : 0.0f; im[i] = 0.0f; }
  fft_radix2(re, im, FEAT_FFT_N);
  float total = 0.0f, pmax = -1.0f; int amax = 0;
  for (int i = 0; i < FEAT_FFT_N / 2; i++) {
    float p = re[i] * re[i] + im[i] * im[i];
    pw[i] = p; total += p;
    if (p > pmax) { pmax = p; amax = i; }
  }
  float ent = 0.0f;
  if (total > 1e-12f)
    for (int i = 0; i < FEAT_FFT_N / 2; i++) { float pn = pw[i] / total; if (pn > 1e-12f) ent -= pn * logf(pn); }
  sink += amax + ent;
  (void)sink;
}

// average us/beat over n_beats consecutive raw beats
static float feature_bench_us(const float *beats, int n_beats, int len) {
  uint32_t t0 = micros();
  for (int i = 0; i < n_beats; i++) feature_extract_one(beats + (size_t)i * len, len);
  return (float)(micros() - t0) / n_beats;
}
