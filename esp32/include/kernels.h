// Config struct layouts shared by the generated model_*.h tables and the kernels
// in main.cpp. Field order MUST match the table-row initializers emitted by
// tools/export_esp32.py.
#pragma once

typedef struct {
  int n_trees, n_feat;
  const int *node_off, *feature, *left, *right;   // node ids are local to each tree
  const float *thr, *leaf_p1;
} RFCfg;

// Gradient-boosted trees. Same node layout as RFCfg, but the traversal differs in two
// ways that BOTH matter for parity: XGBoost splits on strict `<` (sklearn uses `<=`),
// and leaves hold raw logits that are SUMMED onto base_logit (the RF averages leaf
// probabilities). Kept a separate struct so the RF kernel stays branch-free.
typedef struct {
  int n_trees, n_feat;
  const int *node_off, *feature, *left, *right;   // node ids are local to each tree
  const float *thr, *leaf;                        // leaf = logit contribution
  float base_logit;                               // logit(base_score)
} XGBCfg;

typedef struct {
  int kind, n_feat;                  // kind: 0 = linear+calibration, 1 = RBF
  const float *mean, *scale;         // StandardScaler
  int n_members;                     // linear: # calibrated members
  const float *coef, *intercept, *cal_a, *cal_b;
  int n_sv;                          // rbf: # support vectors
  const float *sv, *dual;
  float gamma, rbf_intercept;
} SVMCfg;

typedef struct {
  int n_conv, k, pad, in_len;
  const int *in_ch, *out_ch, *w_off, *b_off;
  const float *conv_w, *conv_b;      // concatenated, BN folded
  int fc_hidden, flat_len, last_c;   // fc_hidden == 0 => GAP head
  const float *fc1_w, *fc1_b, *fc2_w, *fc2_b;
} CNNCfg;

typedef struct {
  int n_layers, hidden, seq_len;
  const int *wih_off, *whh_off;
  const float *wih, *whh, *bih, *bhh, *head_w, *head_b;
} LSTMCfg;

// CNN-LSTM hybrid: the conv stack of CNNCfg (BN folded, same kernel) feeding a 1-layer
// LSTM. The recurrent part runs over `seq_len` = in_len >> n_conv timesteps whose input
// vector is the `lstm_in` conv channels at that timestep — note the conv activations are
// stored CHANNEL-major, so the kernel gathers x_t with a stride of seq_len.
typedef struct {
  int n_conv, k, pad, in_len;
  const int *in_ch, *out_ch, *w_off, *b_off;
  const float *conv_w, *conv_b;
  int seq_len, lstm_in, hidden;
  const float *wih, *whh, *bih, *bhh, *head_w, *head_b;
} CRNNCfg;
