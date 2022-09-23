#include "common.h"

class LHE_KMS {
public:
  LHE_KMS();
  virtual ~LHE_KMS(){};
  // poly_modulus_degree = 4096 ~ 32768
  // sec_level_type::tc128 tc192 tc256
  void generate(COMPUTE_MODE mode, size_t poly_modulus = 16384,
                sec_level_type sec_level = sec_level_type::tc128);
  void destroy(){};

  stringstream getEncryptionParamters() const;
  stringstream getPublicKey() const;
  stringstream getRelinearKey() const;
  stringstream getSecretKey() const;

private:
  COMPUTE_MODE compute_mode_;
  size_t poly_modulus_degree_;
  sec_level_type sec_level_;
  EncryptionParameters params_;
  unique_ptr<KeyGenerator> keygen_;
};