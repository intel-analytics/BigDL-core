#include "ckks_common.h"

double scale = pow(2.0, 40);

CKKS_Common::CKKS_Common(vector<string> secret) {
  stringstream ss;

  // params_
  // relin
  for (const string &s : secret)
    ss << s;
  cout << "Init CKKS_Common stringstream" << endl;
  params_ = make_unique<EncryptionParameters>();
  params_->load(ss);

  context_ = new SEALContext(*params_);
  encoder_ = new CKKSEncoder(*context_);
  evaluator_ = make_unique<Evaluator>(*context_);

  print_line(__LINE__);
  cout << "Get CKKS_Common encryption parameters and print" << endl;
  print_parameters(*context_);

  rln_key_ = make_unique<RelinKeys>();
  rln_key_->load(*context_, ss);
}

Ciphertext CKKS_Common::computeSamplePolynomial(const Ciphertext &x) {
  PolyPlainCoeffs model;
  encoder_->encode(3.14159265, scale, model.coeff3);
  encoder_->encode(0.4, scale, model.coeff1);
  encoder_->encode(1.0, scale, model.coeff0);

  print_line(__LINE__);
  cout << "Compute x^2 and relinearize:" << endl;
  Ciphertext x3;
  evaluator_->square(x, x3);
  evaluator_->relinearize_inplace(x3, *rln_key_);
  cout << "    + Scale of x^2 before rescale: " << log2(x3.scale()) << "bits"
       << endl;

  print_line(__LINE__);
  cout << "Rescale x^2." << endl;
  evaluator_->rescale_to_next_inplace(x3);
  cout << "    + Scale of x^2 after  rescale: " << log2(x3.scale()) << "bits"
       << endl;

  print_line(__LINE__);
  cout << "Compute and rescale PI*x." << endl;
  Ciphertext x1_coeff3;
  evaluator_->multiply_plain(x, model.coeff3, x1_coeff3);
  cout << "    + Scale of PI*x before rescale: " << log2(x1_coeff3.scale())
       << "bits" << endl;
  evaluator_->rescale_to_next_inplace(x1_coeff3);
  cout << "    + Scale of PI*x after rescale: " << log2(x1_coeff3.scale())
       << "bits" << endl;

  print_line(__LINE__);
  cout << "Compute, relinearize, and rescale (PI*x)*x^2." << endl;
  evaluator_->multiply_inplace(x3, x1_coeff3);
  evaluator_->relinearize_inplace(x3, *rln_key_);
  cout << "    + Scale of PI*x^3 before rescale: " << log2(x3.scale())
       << " bits" << endl;
  evaluator_->rescale_to_next_inplace(x3);
  cout << "    + Scale of PI*x^3 after rescale: " << log2(x3.scale()) << " bits"
       << endl;

  print_line(__LINE__);
  cout << "Compute and rescale 0.4*x." << endl;
  Ciphertext x1;
  evaluator_->multiply_plain(x, model.coeff1, x1);
  cout << "    + Scale of 0.4*x before rescale: " << log2(x1.scale()) << " bits"
       << endl;
  evaluator_->rescale_to_next_inplace(x1);
  cout << "    + Scale of 0.4*x after rescale: " << log2(x1.scale()) << " bits"
       << endl;

  cout << endl;
  print_line(__LINE__);
  cout << "Parameters used by all three terms are different." << endl;
  cout << "    + Modulus chain index for x3: "
       << context_->get_context_data(x3.parms_id())->chain_index() << endl;
  cout << "    + Modulus chain index for x1: "
       << context_->get_context_data(x1.parms_id())->chain_index() << endl;
  cout << "    + Modulus chain index for coeff0: "
       << context_->get_context_data(model.coeff0.parms_id())->chain_index()
       << endl;
  cout << endl;

  print_line(__LINE__);
  cout << "The exact scales of all three terms are different:" << endl;
  ios old_fmt(nullptr);
  old_fmt.copyfmt(cout);
  cout << fixed << setprecision(10);
  cout << "    + Exact scale in PI*x^3: " << x3.scale() << endl;
  cout << "    + Exact scale in  0.4*x: " << x1.scale() << endl;
  cout << "    + Exact scale in      1: " << model.coeff0.scale() << endl;
  cout << endl;
  cout.copyfmt(old_fmt);

  print_line(__LINE__);
  cout << "Normalize scales to 2^40." << endl;
  x3.scale() = pow(2.0, 40);
  x1.scale() = pow(2.0, 40);

  print_line(__LINE__);
  cout << "Normalize encryption parameters to the lowest level." << endl;
  parms_id_type last_parms_id = x3.parms_id();
  evaluator_->mod_switch_to_inplace(x1, last_parms_id);
  evaluator_->mod_switch_to_inplace(model.coeff0, last_parms_id);

  print_line(__LINE__);
  cout << "Compute PI*x^3 + 0.4*x + 1." << endl;
  Ciphertext result;
  evaluator_->add(x3, x1, result);
  evaluator_->add_plain_inplace(result, model.coeff0);

  return result;
}

Ciphertext CKKS_Common::computeCAddTable(const vector<Ciphertext> &x) {
  Ciphertext sum;
  evaluator_->add_many(x, sum);
  return sum;
}

Ciphertext CKKS_Common::computeSigmoid3rdDegree(const Ciphertext &x) {
  PolyPlainCoeffs sigmoid;
  encoder_->encode(-0.001593, scale, sigmoid.coeff3);
  encoder_->encode(0.15012, scale, sigmoid.coeff1);
  encoder_->encode(0.5, scale, sigmoid.coeff0);

  Ciphertext x3;
  evaluator_->square(x, x3);
  evaluator_->relinearize_inplace(x3, *rln_key_);
  evaluator_->rescale_to_next_inplace(x3);

  Ciphertext x1_coeff3;
  evaluator_->multiply_plain(x, sigmoid.coeff3, x1_coeff3);
  evaluator_->rescale_to_next_inplace(x1_coeff3);

  evaluator_->multiply_inplace(x3, x1_coeff3);
  evaluator_->relinearize_inplace(x3, *rln_key_);
  evaluator_->rescale_to_next_inplace(x3);

  Ciphertext x1;
  evaluator_->multiply_plain(x, sigmoid.coeff1, x1);
  evaluator_->rescale_to_next_inplace(x1);

  x3.scale() = pow(2.0, 40);
  x1.scale() = pow(2.0, 40);

  parms_id_type last_parms_id = x3.parms_id();
  evaluator_->mod_switch_to_inplace(x1, last_parms_id);
  evaluator_->mod_switch_to_inplace(sigmoid.coeff0, last_parms_id);

  Ciphertext result;
  evaluator_->add(x3, x1, result);
  evaluator_->add_plain_inplace(result, sigmoid.coeff0);

  return result;
}

Ciphertext CKKS_Common::computeSigmoid7thDegree(const Ciphertext &x) {
  PolyPlainCoeffs sigmoid;
  encoder_->encode(0.0000011956, scale, sigmoid.coeff7);
  encoder_->encode(0.0001658331, scale, sigmoid.coeff5);
  encoder_->encode(-0.00819154, scale, sigmoid.coeff3);
  encoder_->encode(0.21687, scale, sigmoid.coeff1);
  encoder_->encode(0.5, scale, sigmoid.coeff0);

  // coeff7 * x^7 + coeff5 * x^5 + coeff3 * x^3 + coeff1 * x + coeff0
  // x: Lv.3, scale 2^40
  // coeff_modulus: P_0, P_1, P_2, P_3, P_4

  // coeff7 * x^7 = coeff7 * x^3 * x^4 = ((coeff7 * x) * x^2) * (x^2 * x^2)
  Ciphertext x2;
  evaluator_->square(x, x2);
  evaluator_->relinearize_inplace(x2, *rln_key_);
  evaluator_->rescale_to_next_inplace(x2); // Lv.2, scale 2^80 / P_3

  Ciphertext x4;
  evaluator_->square(x2, x4);
  evaluator_->relinearize_inplace(x4, *rln_key_);
  evaluator_->rescale_to_next_inplace(x4); // Lv.1, scale (2^80/P_3)^2 / P_2

  Ciphertext x7_coeff7;
  evaluator_->multiply_plain(x, sigmoid.coeff7, x7_coeff7);
  evaluator_->rescale_to_next_inplace(
      x7_coeff7); // coeff7 * x, Lv.2, scale 2^80 / P_3

  evaluator_->multiply_inplace(x7_coeff7, x2);
  evaluator_->relinearize_inplace(x7_coeff7, *rln_key_);
  evaluator_->rescale_to_next_inplace(
      x7_coeff7); // coeff7 * x^3, Lv.1, scale (2^80/P_3)^2 / P_2

  evaluator_->multiply_inplace(x7_coeff7, x4);
  evaluator_->relinearize_inplace(x7_coeff7, *rln_key_);
  evaluator_->rescale_to_next_inplace(
      x7_coeff7); // coeff7 * x^7, Lv.0, scale ((2^80/P_3)^2/P_2)^2 / P_1

  // coeff5 * x^5 = coeff5 * x^2 * x^3 = (coeff5 * x^2) * (x * x^2)
  Ciphertext x3;
  evaluator_->mod_switch_to_next(x, x3); // x, Lv.2, scale 2^40
  evaluator_->multiply_inplace(x3, x2);
  evaluator_->relinearize_inplace(x3, *rln_key_);
  evaluator_->rescale_to_next_inplace(x3); // x^3, Lv.1, scale 2^120 / P_3 / P_2

  Plaintext coeff5;
  evaluator_->mod_switch_to_next(sigmoid.coeff5, coeff5); // Lv.2, scale 2^40

  Ciphertext x5_coeff5;
  evaluator_->multiply_plain(x2, coeff5, x5_coeff5);
  evaluator_->rescale_to_next_inplace(
      x5_coeff5); // coeff5 * x^2, Lv.1, scale 2^120 / P_3 / P_2

  evaluator_->multiply_inplace(x5_coeff5, x3);
  evaluator_->relinearize_inplace(x5_coeff5, *rln_key_);
  evaluator_->rescale_to_next_inplace(
      x5_coeff5); // coeff5 * x^5, Lv.0, scale (2^120/P_3/P_2)^2 / P_1

  // coeff3 * x^3 = coeff3 * x * x^2
  Ciphertext x3_coeff3;
  evaluator_->multiply_plain(x, sigmoid.coeff3, x3_coeff3);
  evaluator_->rescale_to_next_inplace(
      x3_coeff3); // coeff3 * x, Lv.2, scale 2^80 / P_3

  evaluator_->multiply_inplace(x3_coeff3, x2);
  evaluator_->relinearize_inplace(x3_coeff3, *rln_key_);
  evaluator_->rescale_to_next_inplace(
      x3_coeff3); // coeff3 * x^3, Lv.1, scale (2^80/P_3)^2 / P_2

  Ciphertext x1_coeff1;
  evaluator_->multiply_plain(x, sigmoid.coeff1, x1_coeff1);
  evaluator_->rescale_to_next_inplace(
      x1_coeff1); // coeff1 * x, Lv.2, scale 2^80 / P_3

  x7_coeff7.scale() = pow(2.0, 40);
  x5_coeff5.scale() = pow(2.0, 40);
  x3_coeff3.scale() = pow(2.0, 40);
  x1_coeff1.scale() = pow(2.0, 40);

  parms_id_type last_parms_id = x7_coeff7.parms_id();
  evaluator_->mod_switch_to_inplace(x3_coeff3, last_parms_id);
  evaluator_->mod_switch_to_inplace(x1_coeff1, last_parms_id);
  evaluator_->mod_switch_to_inplace(sigmoid.coeff0, last_parms_id);

  Ciphertext result;
  evaluator_->add_many({x7_coeff7, x5_coeff5, x3_coeff3, x1_coeff1}, result);
  evaluator_->add_plain_inplace(result, sigmoid.coeff0);

  return result;
}

Ciphertext CKKS_Common::computeLogarithm3rdDegree(const Ciphertext &x,
                                                  uint32_t base) {
  double base_change = 1.0;
  if (base >= 2) {
    base_change = 1 / log(base);
  }

  PolyPlainCoeffs ln;
  encoder_->encode(0.33333 * base_change, scale, ln.coeff3);
  encoder_->encode(-1.5 * base_change, scale, ln.coeff2);
  encoder_->encode(3.0 * base_change, scale, ln.coeff1);
  encoder_->encode(-1.83333 * base_change, scale, ln.coeff0);

  // make coeffs and x in same level, Lv.n, and scale
  // Ciphertext xx = x;
  // xx.scale() = pow(2.0, 40);
  parms_id_type parms_id = x.parms_id();
  evaluator_->mod_switch_to_inplace(ln.coeff3, parms_id);
  evaluator_->mod_switch_to_inplace(ln.coeff2, parms_id);
  evaluator_->mod_switch_to_inplace(ln.coeff1, parms_id);
  evaluator_->mod_switch_to_inplace(ln.coeff0, parms_id);

  Ciphertext x1_coeff1;
  evaluator_->multiply_plain(x, ln.coeff1, x1_coeff1);
  evaluator_->rescale_to_next_inplace(x1_coeff1); // Lv.n-1

  Ciphertext x2;
  evaluator_->square(x, x2);
  evaluator_->relinearize_inplace(x2, *rln_key_);
  evaluator_->rescale_to_next_inplace(x2); // Lv.n-1

  Ciphertext x1;
  evaluator_->mod_switch_to_next(x, x1);

  Ciphertext x2_coeff2;
  evaluator_->multiply_plain(x, ln.coeff2, x2_coeff2);
  evaluator_->rescale_to_next_inplace(x2_coeff2); // coeff2*x, Lv.n-1

  evaluator_->multiply_inplace(x2_coeff2, x1); // coeff2*x^2
  evaluator_->relinearize_inplace(x2_coeff2, *rln_key_);
  evaluator_->rescale_to_next_inplace(x2_coeff2); // Lv.n-2

  Ciphertext x3_coeff3;
  evaluator_->multiply_plain(x, ln.coeff3, x3_coeff3);
  evaluator_->rescale_to_next_inplace(x3_coeff3); // Lv.n-1

  evaluator_->multiply_inplace(x3_coeff3, x2);
  evaluator_->relinearize_inplace(x3_coeff3, *rln_key_);
  evaluator_->rescale_to_next_inplace(x3_coeff3); // Lv.n-2

  x3_coeff3.scale() = pow(2.0, 40);
  x2_coeff2.scale() = pow(2.0, 40);
  x1_coeff1.scale() = pow(2.0, 40);

  parms_id_type last_parms_id = x3_coeff3.parms_id();
  evaluator_->mod_switch_to_inplace(x2_coeff2, last_parms_id);
  evaluator_->mod_switch_to_inplace(x1_coeff1, last_parms_id);
  evaluator_->mod_switch_to_inplace(ln.coeff0, last_parms_id);

  Ciphertext result;
  evaluator_->add_many({x3_coeff3, x2_coeff2, x1_coeff1}, result);
  evaluator_->add_plain_inplace(result, ln.coeff0);

  return result;
}

Ciphertext CKKS_Common::computeBCE(const Ciphertext &x, const Ciphertext &y) {
  // -y * log(x) - (1-y) * log(1-x) = -y * log(x) + (y-1) * log(1-x)
  Plaintext pos_one, neg_one;
  encoder_->encode(1.0, scale, pos_one);
  encoder_->encode(-1.0, scale, neg_one);

  Ciphertext log_x = computeLogarithm3rdDegree(x, 0); // Lv.n-2
  parms_id_type parms_id = log_x.parms_id();

  Ciphertext neg_y;
  evaluator_->multiply_plain(y, neg_one, neg_y);
  evaluator_->rescale_to_next_inplace(neg_y);         // Lv.n-1
  evaluator_->mod_switch_to_inplace(neg_y, parms_id); // Lv.n-2

  Ciphertext neg_y_mul_log_x;
  evaluator_->multiply(neg_y, log_x, neg_y_mul_log_x);
  evaluator_->relinearize_inplace(neg_y_mul_log_x, *rln_key_);
  evaluator_->rescale_to_next_inplace(neg_y_mul_log_x); // Lv.n-3

  Ciphertext y_sub_one;
  evaluator_->sub_plain(y, pos_one, y_sub_one); // Lv.n

  parms_id = x.parms_id();
  evaluator_->mod_switch_to_inplace(neg_one, parms_id);

  Ciphertext neg_x;
  evaluator_->multiply_plain(x, neg_one, neg_x);
  evaluator_->rescale_to_next_inplace(neg_x); // Lv.n-1

  parms_id = neg_x.parms_id();
  evaluator_->mod_switch_to_inplace(pos_one, parms_id);

  Ciphertext one_sub_x;
  neg_x.scale() = pow(2.0, 40);
  evaluator_->add_plain(neg_x, pos_one, one_sub_x); // Lv.n-1

  Ciphertext log_one_sub_x = computeLogarithm3rdDegree(one_sub_x, 0); // Lv.n-3
  parms_id = log_one_sub_x.parms_id();
  evaluator_->mod_switch_to_inplace(y_sub_one, parms_id);

  Ciphertext y_sub_one_mul_log;
  evaluator_->multiply(y_sub_one, log_one_sub_x, y_sub_one_mul_log);
  evaluator_->relinearize_inplace(y_sub_one_mul_log, *rln_key_);
  evaluator_->rescale_to_next_inplace(y_sub_one_mul_log); // Lv.n-4

  y_sub_one_mul_log.scale() = pow(2.0, 40);
  neg_y_mul_log_x.scale() = pow(2.0, 40);
  parms_id = y_sub_one_mul_log.parms_id();
  evaluator_->mod_switch_to_inplace(neg_y_mul_log_x, parms_id);

  Ciphertext result;
  evaluator_->add(neg_y_mul_log_x, y_sub_one_mul_log, result); // Lv.n-4

  return result;
}

Ciphertext CKKS_Common::computeBackwards(const Ciphertext &y_hat,
                                         const Ciphertext &y) {
  Ciphertext backwards;
  Ciphertext yy;

  parms_id_type parms_id = y_hat.parms_id();
  evaluator_->mod_switch_to(y, parms_id, yy);
  evaluator_->sub(y_hat, yy, backwards);

  return backwards;
}
