#pragma once

#include "common.h"
#include <numeric>
#include <random>

class DataEncryptor {
public:
  DataEncryptor(vector<string> secret);
  DataEncryptor(string secret);
  virtual ~DataEncryptor() = default;

  stringstream encrypt(const vector<double> &x) const;
  vector<double> decrypt(const Ciphertext &encrypted_result) const;

  SEALContext* context_;
  CKKSEncoder* encoder_;

private:
  unique_ptr<EncryptionParameters> params_;
  unique_ptr<PublicKey> pub_key_;
  unique_ptr<RelinKeys> rln_key_;
  unique_ptr<SecretKey> sec_key_;
  unique_ptr<Encryptor> encryptor_;
  unique_ptr<Decryptor> decryptor_;
};
