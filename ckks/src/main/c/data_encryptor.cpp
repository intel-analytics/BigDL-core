#include "data_encryptor.h"

DataEncryptor::DataEncryptor(vector<string> secret) {
  stringstream ss;

  for (const string &s : secret)
    ss << s;
  cout << "Init stringstream" << endl;

  params_ = make_unique<EncryptionParameters>();
  params_->load(ss);
  context_ = new SEALContext(*params_);
  print_line(__LINE__);
  cout << "Get encryption parameters and print" << endl;
  print_parameters(*context_);

  pub_key_ = make_unique<PublicKey>();
  pub_key_->load(*context_, ss);

  rln_key_ = make_unique<RelinKeys>();
  rln_key_->load(*context_, ss);

  sec_key_ = make_unique<SecretKey>();
  sec_key_->load(*context_, ss);

  encoder_ = new CKKSEncoder(*context_);
  encryptor_ = make_unique<Encryptor>(*context_, *pub_key_);

  cout << "Init DataEncryptor" << endl;
  if (sec_key_)
    decryptor_ = make_unique<Decryptor>(*context_, *sec_key_);
  cout << "Init decryptor_" << endl;
}

stringstream DataEncryptor::encrypt(const vector<double> &x) const {
  Plaintext x_plain;
  double scale = pow(2.0, 40);
  encoder_->encode(x, scale, x_plain);

  stringstream ciphertext_stream;
  Serializable<Ciphertext> x_encrypted = encryptor_->encrypt(x_plain);
  auto encrypted_size = x_encrypted.save(ciphertext_stream);
  return ciphertext_stream;
}

vector<double>
DataEncryptor::decrypt(const Ciphertext &encrypted_result) const {
  Plaintext plain_result;
  decryptor_->decrypt(encrypted_result, plain_result);

  vector<double> result;
  encoder_->decode(plain_result, result);
  return result;
}

