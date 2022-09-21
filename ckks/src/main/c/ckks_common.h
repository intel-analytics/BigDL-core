#pragma once

#include "common.h"

#include <algorithm>

class CKKS_Common {
public:
  CKKS_Common(vector<string> secret);
  CKKS_Common(string secret);

  Ciphertext computeSamplePolynomial(const Ciphertext &x);
  Ciphertext computeCAddTable(const vector<Ciphertext> &x);
  Ciphertext computeSigmoid3rdDegree(const Ciphertext &x);
  Ciphertext computeSigmoid7thDegree(const Ciphertext &x);
  Ciphertext computeLogarithm3rdDegree(const Ciphertext &x, uint32_t base);
  Ciphertext computeBCE(const Ciphertext &x, const Ciphertext &y);
  Ciphertext computeBackwards(const Ciphertext &y_hat, const Ciphertext &y);

  SEALContext* context_;
  CKKSEncoder* encoder_;

private:
  unique_ptr<EncryptionParameters> params_;
  unique_ptr<Evaluator> evaluator_;
  unique_ptr<RelinKeys> rln_key_;
};
