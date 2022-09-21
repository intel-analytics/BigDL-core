#pragma once

#include "common.h"
#include "ckks_common.h"
#include "data_encryptor.h"
// #include <numeric>
// #include <random>


class LogisticRegression {
public:
  LogisticRegression() {
  }
  virtual ~LogisticRegression() = default;

  string inference_chunk(const string &x, uint32_t chunk_id);
  TrainFeedbacks train_chunk(const string &x, const string &y, uint32_t chunk_id);

  void init(vector<string> secret) {
    encryptor_ = new DataEncryptor(secret);
    ckks_common_ = new CKKS_Common(secret);
    cout << "Init DataEncryptor & CKKS_Common" << endl;
    context_ = encryptor_->context_;
    encoder_ = encryptor_->encoder_;
  }

  void set_learning_rate(double learning_rate) {learning_rate_ = learning_rate;}

  void set_batch_size(uint32_t batch_size) {batch_size_ = batch_size;}

  void set_max_iter(uint32_t max_iter) {max_iter_ = max_iter;}

  void set_data(vector<vector<double>> training_set, vector<vector<double>> eval_set) {
    training_set_ = training_set;
    eval_set_ = eval_set;
    feature_size_ = training_set_[0].size() - 1;

    weights_.resize(feature_size_ + 1);

    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(1e-8, 1.0);

    double t;
    for (size_t i = 0; i < weights_.size(); i++) {
      t = dist(mt);
      weights_[i] = t;
    }

    w1_ = weights_;
  }

  void train();

  vector<double> local_forward(vector<vector<double>> input) {
    assert(input[0].size() == weights_.size() - 1);

    vector<double> forwards;
    for (auto &sample : input) {
      double output = weights_[0]; // bias
      for (size_t i = 0; i < feature_size_; i++) {
        output += sample[i] * weights_[i + 1];
      }

      output = max(-4.0, min(output, 4.0));

      forwards.push_back(output);
    }

    return forwards;
  }

  vector<string> preprocess(vector<double> input);

  void weight_update(vector<double> backwards, vector<vector<double>> x,
                     uint32_t iter) {
    assert(backwards.size() == x.size());
    assert(x[0].size() == weights_.size() - 1);

    double learning_rate = 10.0 / (iter + 2.0);
    double new_lambda = (sqrt(pow(lambda_, 2) * 4.0 + 1.0) + 1.0) / 2.0;
    double smoothing = (1.0 - lambda_) / new_lambda;
    lambda_ = new_lambda;

    for (size_t i = 0; i < weights_.size(); i++) {
      double dw = 0;
      if (i == 0) {
        dw = accumulate(backwards.begin(), backwards.end(), 0.0) /
             backwards.size();
      } else {
        for (size_t idx = 0; idx < backwards.size(); idx++) {
          dw += backwards[idx] * x[idx][i - 1];
        }
        dw /= backwards.size();
      }

      double new_w1 = weights_[i] - learning_rate_ * dw;
      double new_weight = (1 - smoothing) * new_w1 + smoothing * w1_[i];

      weights_[i] = new_weight;
      w1_[i] = new_w1;
    }
  }

private:
  uint32_t feature_size_;
  vector<vector<double>> training_set_;
  vector<vector<double>> eval_set_;
  vector<double> weights_;
  vector<double> w1_;
  DataEncryptor* encryptor_;
  CKKSEncoder* encoder_;
  SEALContext* context_;
  CKKS_Common* ckks_common_;
  double learning_rate_ = 0.01;
  uint32_t batch_size_ = 1000U;
  uint32_t max_iter_ = 300U;

  double lambda_ = 0;
};
