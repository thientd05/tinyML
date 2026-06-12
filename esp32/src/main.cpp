// ESP32-WROOM-32 benchmark: every ECG model in the capacity sweep (4 families x ~6
// points) in one firmware, built from the generated model table in test_data.h.
//
// On boot it runs every model over the small embedded DS2 sample set and prints a table.
// With N small the per-model acc/prec/recall/f1/auc are only a sanity check — trustworthy
// quality is the full-DS2 metrics on PC (parity == 1.0000). What the device uniquely
// provides is the per-beat LATENCY and peak heap, which feed the feasibility gate.
//   columns: accuracy, precision, recall, F1, AUC, parity (vs PC, expected 1.0000),
//   us/beat, per-model peak heap.
//
// The generic C kernels mirror the NumPy reproductions in tools/export_esp32.py,
// which are asserted equal to sklearn/torch at export time.

#include <Arduino.h>
#include <math.h>
#include "kernels.h"
#include "test_data.h"
#include "model_rf.h"
#include "model_svm.h"
#include "model_cnn.h"
#include "model_lstm.h"
#include "feature_bench.h"

static inline float sigmoidf(float z) { return 1.0f / (1.0f + expf(-z)); }

// scratch buffers (sized by the generators to the largest model)
static float cnnA[CNN_BUF], cnnB[CNN_BUF];
static float lstmA[LSTM_BUF], lstmB[LSTM_BUF];

// ---------------- Random Forest ----------------
static int rf_infer(const RFCfg &m, const float *x, float *score) {
  float p1 = 0.0f;
  for (int tr = 0; tr < m.n_trees; tr++) {
    int off = m.node_off[tr], nd = 0;
    while (m.left[off + nd] != -1)
      nd = (x[m.feature[off + nd]] <= m.thr[off + nd]) ? m.left[off + nd] : m.right[off + nd];
    p1 += m.leaf_p1[off + nd];
  }
  p1 /= m.n_trees;
  *score = p1;
  return p1 > 0.5f ? 1 : 0;
}

// ---------------- SVM (linear+calibration | RBF) ----------------
static int svm_infer(const SVMCfg &m, const float *x, float *score) {
  float xs[N_FEAT];
  for (int i = 0; i < m.n_feat; i++) xs[i] = (x[i] - m.mean[i]) / m.scale[i];
  if (m.kind == 1) {                                  // RBF
    float dec = m.rbf_intercept;
    for (int j = 0; j < m.n_sv; j++) {
      const float *sv = m.sv + (size_t)j * m.n_feat;
      float d2 = 0.0f;
      for (int i = 0; i < m.n_feat; i++) { float d = xs[i] - sv[i]; d2 += d * d; }
      dec += m.dual[j] * expf(-m.gamma * d2);
    }
    *score = dec;
    return dec > 0.0f ? 1 : 0;
  }
  float p = 0.0f;                                     // linear + sigmoid calibration
  for (int mm = 0; mm < m.n_members; mm++) {
    float d = m.intercept[mm];
    for (int i = 0; i < m.n_feat; i++) d += m.coef[mm * m.n_feat + i] * xs[i];
    p += sigmoidf(-(m.cal_a[mm] * d + m.cal_b[mm]));
  }
  p /= m.n_members;
  *score = p;
  return p >= 0.5f ? 1 : 0;
}

// ---------------- 1D CNN (generic depth, GAP or FC head) ----------------
static int cnn_infer(const CNNCfg &m, const float *x, float *score) {
  const float *src = x;
  int sc_ch = 1, sc_len = m.in_len;
  float *dst = cnnA;
  for (int l = 0; l < m.n_conv; l++) {
    int oc = m.out_ch[l], ic = m.in_ch[l], out_len = sc_len / 2;
    const float *w = m.conv_w + m.w_off[l];
    const float *b = m.conv_b + m.b_off[l];
    for (int c = 0; c < oc; c++) {
      for (int p = 0; p < out_len; p++) {
        float best = -INFINITY;
        for (int q = 0; q < 2; q++) {                 // maxpool over the pair (2p, 2p+1)
          int t = 2 * p + q;
          float acc = b[c];
          for (int i = 0; i < ic; i++)
            for (int kk = 0; kk < m.k; kk++) {
              int idx = t + kk - m.pad;
              if (idx >= 0 && idx < sc_len)
                acc += w[(c * ic + i) * m.k + kk] * src[i * sc_len + idx];
            }
          if (acc < 0.0f) acc = 0.0f;                 // relu
          if (acc > best) best = acc;
        }
        dst[c * out_len + p] = best;
      }
    }
    src = dst; sc_ch = oc; sc_len = out_len;
    dst = (dst == cnnA) ? cnnB : cnnA;
  }
  float l0, l1;
  if (m.fc_hidden == 0) {                             // GAP -> linear
    float g[64];
    for (int c = 0; c < sc_ch; c++) {
      float s = 0.0f;
      for (int t = 0; t < sc_len; t++) s += src[c * sc_len + t];
      g[c] = s / sc_len;
    }
    l0 = m.fc2_b[0]; l1 = m.fc2_b[1];
    for (int c = 0; c < sc_ch; c++) { l0 += m.fc2_w[c] * g[c]; l1 += m.fc2_w[m.last_c + c] * g[c]; }
  } else {                                            // flatten -> fc -> relu -> linear
    float h[64];
    for (int o = 0; o < m.fc_hidden; o++) {
      float a = m.fc1_b[o];
      for (int f = 0; f < m.flat_len; f++) a += m.fc1_w[o * m.flat_len + f] * src[f];
      h[o] = a > 0.0f ? a : 0.0f;
    }
    l0 = m.fc2_b[0]; l1 = m.fc2_b[1];
    for (int o = 0; o < m.fc_hidden; o++) { l0 += m.fc2_w[o] * h[o]; l1 += m.fc2_w[m.fc_hidden + o] * h[o]; }
  }
  *score = l1 - l0;
  return l1 > l0 ? 1 : 0;
}

// ---------------- LSTM (generic #layers, gate order i,f,g,o) ----------------
static int lstm_infer(const LSTMCfg &m, const float *seq, float *score) {
  const int H = m.hidden, T = m.seq_len;
  float h[64], c[64];
  const float *inbuf = nullptr;     // hidden seq from previous layer (l>0)
  float *dst = lstmA;
  for (int l = 0; l < m.n_layers; l++) {
    int insz = (l == 0) ? 1 : H;
    const float *wih = m.wih + m.wih_off[l];
    const float *whh = m.whh + m.whh_off[l];
    const float *bih = m.bih + l * 4 * H;
    const float *bhh = m.bhh + l * 4 * H;
    for (int j = 0; j < H; j++) { h[j] = 0.0f; c[j] = 0.0f; }
    for (int t = 0; t < T; t++) {
      const float *xt = (l == 0) ? &seq[t] : &inbuf[t * H];
      for (int j = 0; j < H; j++) {
        int ri = j, rf = H + j, rg = 2 * H + j, ro = 3 * H + j;
        float gi = bih[ri] + bhh[ri], gf = bih[rf] + bhh[rf];
        float gg = bih[rg] + bhh[rg], go = bih[ro] + bhh[ro];
        for (int cc = 0; cc < insz; cc++) {
          float xv = xt[cc];
          gi += wih[ri * insz + cc] * xv; gf += wih[rf * insz + cc] * xv;
          gg += wih[rg * insz + cc] * xv; go += wih[ro * insz + cc] * xv;
        }
        for (int kk = 0; kk < H; kk++) {
          float hv = h[kk];
          gi += whh[ri * H + kk] * hv; gf += whh[rf * H + kk] * hv;
          gg += whh[rg * H + kk] * hv; go += whh[ro * H + kk] * hv;
        }
        float nc = sigmoidf(gf) * c[j] + sigmoidf(gi) * tanhf(gg);
        dst[t * H + j] = sigmoidf(go) * tanhf(nc);    // store; commit after the j-loop
        c[j] = nc;
      }
      for (int j = 0; j < H; j++) h[j] = dst[t * H + j];
    }
    inbuf = dst; dst = (dst == lstmA) ? lstmB : lstmA;
  }
  const float *last = inbuf + (size_t)(T - 1) * H;
  float l0 = m.head_b[0], l1 = m.head_b[1];
  for (int k = 0; k < H; k++) { l0 += m.head_w[k] * last[k]; l1 += m.head_w[H + k] * last[k]; }
  *score = l1 - l0;
  return l1 > l0 ? 1 : 0;
}

// ---------------- dispatch + driver ----------------
enum Family { F_RF, F_SVM, F_CNN, F_LSTM };
struct Entry { const char *name; Family fam; int idx; const float *data; int stride; };

static int dispatch(const Entry &e, const float *x, float *score) {
  switch (e.fam) {
    case F_RF:   return rf_infer(RF_CFG[e.idx], x, score);
    case F_SVM:  return svm_infer(SVM_CFG[e.idx], x, score);
    case F_CNN:  return cnn_infer(CNN_CFG[e.idx], x, score);
    default:     return lstm_infer(LSTM_CFG[e.idx], x, score);
  }
}

static float scores[N_SAMPLES];
static int preds[N_SAMPLES];

// AUC via Mann-Whitney U (ties counted as 0.5). O(n^2), trivial for N=400.
static float auc_of(const int *y) {
  long npos = 0, nneg = 0;
  for (int i = 0; i < N_SAMPLES; i++) (y[i] ? npos : nneg)++;
  if (!npos || !nneg) return NAN;
  double u = 0.0;
  for (int i = 0; i < N_SAMPLES; i++) {
    if (!y[i]) continue;
    for (int j = 0; j < N_SAMPLES; j++) {
      if (y[j]) continue;
      if (scores[i] > scores[j]) u += 1.0;
      else if (scores[i] == scores[j]) u += 0.5;
    }
  }
  return (float)(u / ((double)npos * nneg));
}

static void run(const Entry &e, int model_index) {
  uint32_t heap0 = ESP.getFreeHeap();
  uint32_t t_sum = 0, t_max = 0;
  int tp = 0, fp = 0, fn = 0, tn = 0, parity = 0;
  const int *pc = &Y_PC[(size_t)model_index * N_SAMPLES];
  for (int i = 0; i < N_SAMPLES; i++) {
    uint32_t t0 = micros();
    int p = dispatch(e, e.data + (size_t)i * e.stride, &scores[i]);
    uint32_t dt = micros() - t0;
    t_sum += dt; if (dt > t_max) t_max = dt;
    preds[i] = p;
    if (p == Y_TRUE[i]) { if (p) tp++; else tn++; } else { if (p) fp++; else fn++; }
    if (p == pc[i]) parity++;
    if ((i & 31) == 0) yield();   // feed the task watchdog during slow models (svm-rbf / lstm)
  }
  float acc = (float)(tp + tn) / N_SAMPLES;
  float prec = (tp + fp) ? (float)tp / (tp + fp) : 0.0f;
  float rec = (tp + fn) ? (float)tp / (tp + fn) : 0.0f;
  float f1 = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0.0f;
  uint32_t peak = heap0 - ESP.getMinFreeHeap();
  Serial.printf("%-16s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %7.1f | %5u\n",
                e.name, acc, prec, rec, f1, auc_of(Y_TRUE), (float)parity / N_SAMPLES,
                (float)t_sum / N_SAMPLES, peak);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  // Built from the generated model table (test_data.h) so it scales with the sweep.
  Entry models[NUM_MODELS];
  for (int i = 0; i < NUM_MODELS; i++) {
    Family fam = (Family)MODEL_FAM[i];
    int sub = MODEL_SUB[i];
    const float *data; int stride;
    switch (fam) {
      case F_CNN:  data = RAW; stride = CNN_CFG[sub].in_len; break;
      case F_LSTM: data = SEQ; stride = LSTM_CFG[sub].seq_len; break;
      default:     data = FEAT; stride = N_FEAT; break;   // RF / SVM use the 21-D features
    }
    models[i] = {MODEL_NAMES[i], fam, sub, data, stride};
  }

  Serial.println();
  Serial.println(F("================== ESP32-WROOM-32  ECG model sweep benchmark =================="));
  Serial.printf("samples: %d   features: %d   models: %d   free heap: %u B\n",
                N_SAMPLES, N_FEAT, NUM_MODELS, ESP.getFreeHeap());
  Serial.println(F("-------------------------------------------------------------------------------"));
  Serial.println(F("model            | acc    | prec   | recall | f1     | auc    | parity | us/beat | heapB"));
  Serial.println(F("-------------------------------------------------------------------------------"));
  for (int i = 0; i < NUM_MODELS; i++) run(models[i], i);
  Serial.println(F("-------------------------------------------------------------------------------"));
  // feature-extraction front-end cost (db4 wavelet + FFT). Add this to RF/SVM us/beat for
  // a fair cross-family detection-time comparison (pass it to analyze.py --feature-us).
  float feat_us = feature_bench_us(RAW, N_SAMPLES, RAW_LEN);
  Serial.printf("feat_extract (db4+FFT front-end, RF/SVM only): %.1f us/beat\n", feat_us);
  Serial.println(F("==============================================================================="));
  Serial.println(F("parity column should be 1.0000 for every model. done. (loop idle)"));
}

void loop() { delay(10000); }
