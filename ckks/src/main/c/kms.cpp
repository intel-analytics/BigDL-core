#include "kms.h"

LHE_KMS::LHE_KMS() { params_ = EncryptionParameters(scheme_type::ckks); }

void LHE_KMS::generate(COMPUTE_MODE mode, size_t poly_modulus,
                       sec_level_type sec_level) {
  poly_modulus_degree_ = poly_modulus;
  sec_level_ = sec_level;

  params_.set_poly_modulus_degree(poly_modulus_degree_);

  print_line(__LINE__);
  cout << "The upper bound for the total bit-length of the coeff_modulus is "
       << CoeffModulus::MaxBitCount(poly_modulus_degree_, sec_level_) << endl;

  if (mode == COMPUTE_MODE::Inference) {
    params_.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree_, {60, 40, 40, 40, 60})); //{ 60, 40, 40, 60 }));
  } else if (mode == COMPUTE_MODE::Train) {
    params_.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree_,
        {60, 40, 40, 40, 40, 40, 40, 40, 60})); //{ 60, 40, 40, 60 }));
  }

  SEALContext context(params_, true, sec_level_);

  print_line(__LINE__);
  cout << "Set encryption parameters and print" << endl;
  print_parameters(context);

  cout << "Parameter validation (success): "
       << context.parameter_error_message() << endl;

  keygen_ = make_unique<KeyGenerator>(context);
}

stringstream LHE_KMS::getEncryptionParamters() const {
  stringstream params_stream;
  auto size = params_.save(params_stream, compr_mode_type::zstd);

  print_line(__LINE__);
  cout << "EncryptionParameters: wrote " << size << " bytes" << endl;

  return params_stream;
}

stringstream LHE_KMS::getPublicKey() const {
  stringstream public_key_stream;

  Serializable<PublicKey> public_key = keygen_->create_public_key();
  auto pk_size = public_key.save(public_key_stream);

  return public_key_stream;
}

stringstream LHE_KMS::getRelinearKey() const {
  stringstream relin_key_stream;

  Serializable<RelinKeys> relin_key = keygen_->create_relin_keys();
  auto rk_size = relin_key.save(relin_key_stream);

  return relin_key_stream;
}

stringstream LHE_KMS::getSecretKey() const {
  SecretKey secret_key = keygen_->secret_key();

  stringstream secret_key_stream;
  auto sk_size = secret_key.save(secret_key_stream);

  return secret_key_stream;
}
