#pragma once

#include "seal/seal.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cassert>

using namespace std;
using namespace seal;

enum class COMPUTE_MODE { Inference, Train };

struct TrainFeedbacks {
  string loss;
  string backwards;
};

struct PolyPlainCoeffs {
  Plaintext coeff7, coeff5, coeff3, coeff2, coeff1, coeff0;
};

/*
Helper function: Prints a vector of floating-point values.
*/
template <typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4,
                         int prec = 3) {
  /*
  Save the formatting information for std::cout.
  */
  std::ios old_fmt(nullptr);
  old_fmt.copyfmt(std::cout);

  std::size_t slot_count = vec.size();

  std::cout << std::fixed << std::setprecision(prec);
  std::cout << std::endl;
  if (slot_count <= 2 * print_size) {
    std::cout << "    [";
    for (std::size_t i = 0; i < slot_count; i++) {
      std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
    }
  } else {
    vec.resize(std::max(vec.size(), 2 * print_size));
    std::cout << "    [";
    for (std::size_t i = 0; i < print_size; i++) {
      std::cout << " " << vec[i] << ",";
    }
    if (vec.size() > 2 * print_size) {
      std::cout << " ...,";
    }
    for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
      std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
    }
  }
  std::cout << std::endl;

  /*
  Restore the old std::cout formatting.
  */
  std::cout.copyfmt(old_fmt);
}

static void generateInput(vector<double> &input, size_t slot_cnt) {
  input.reserve(slot_cnt);
  double curr_point = 0;
  double step_size = 2.0 / (static_cast<double>(slot_cnt * 2.3) - 1);
  for (size_t i = 0; i < slot_cnt * 2.3; i++) {
    input.push_back(curr_point);
    curr_point += step_size;
  }
}

static double computeLogarithm3rdDegree(const double &x, uint32_t base = 0) {
  double base_change = 1.0;
  if (base >= 2) {
    base_change = 1 / log(base);
  }

  vector<double> coeffs{0.33333, -1.5, 3.0, -1.83333};
  double r =
      coeffs[0] * pow(x, 3) + coeffs[1] * pow(x, 2) + coeffs[2] * x + coeffs[3];

  return base_change * r;
}

static double computeSigmoid7thDegree(const double &x) {
  vector<double> coeffs{0.0000011956, 0.0001658331, -0.00819154, 0.21687, 0.5};
  return coeffs[0] * pow(x, 7) + coeffs[1] * pow(x, 5) + coeffs[2] * pow(x, 3) +
         coeffs[3] * x + coeffs[4];
}

static vector<double> computeBCE(double x, double y) {
  double true_result = (-1) * y * log2(x) - (1 - y) * log2(1 - x);
  double appr_result = (-1) * y * computeLogarithm3rdDegree(x, 2) -
                       (1 - y) * computeLogarithm3rdDegree(1 - x, 2);

  return vector<double>{true_result, appr_result};
}

static vector<uint32_t> generateRandomIdx(uint32_t samples, uint32_t max_idx) {
  assert(samples <= max_idx);

  random_device rd;
  mt19937 mt(rd());
  uniform_int_distribution<uint32_t> dist(1, max_idx);

  vector<uint32_t> idxes;
  while (true) {
    uint32_t i = dist(mt);

    if (find(idxes.begin(), idxes.end(), i) != idxes.end())
      continue;
    else
      idxes.push_back(i);

    if (idxes.size() >= samples)
      break;
  }

  return idxes;
}

static void printExpectedLosses(uint32_t client_num,
                                const vector<double> &input,
                                const vector<double> &target, size_t slot_cnt) {
  assert(input.size() == target.size());

  cout << "    + Expected result (" << client_num << " clients):" << endl;
  vector<double> true_result;
  vector<double> appr_result;

  for (int i = 0; i < input.size(); i++) {
    double x = input[i];
    double y = target[i];
    vector<double> result = computeBCE(x * client_num, y);
    true_result.push_back(result[0]);
    appr_result.push_back(result[1]);

    if (true_result.size() % slot_cnt == 0) {
      print_vector(true_result, 3, 7);
      true_result.clear();

      print_vector(appr_result, 3, 7);
      appr_result.clear();
    }
  }
  print_vector(true_result, 3, 7);
  print_vector(appr_result, 3, 7);

  ofstream output_file;
  output_file.open("../../temp/local_loss.txt", ios::out | ios::trunc);
  ostream_iterator<double> output_iter(output_file, "\n");
  auto iter = copy(true_result.begin(), true_result.end(), output_iter);
  copy(appr_result.begin(), appr_result.end(), iter);
  output_file.close();
}

static double localValidate(const vector<double> &input,
                            const vector<double> &label) {
  double output, predict, gt;
  uint32_t hit = 0, miss = 0;

  for (size_t i = 0; i < input.size(); i++) {
    output = computeSigmoid7thDegree(input[i]);
    predict = (output >= 0.5) ? 1.0 : 0.0;

    gt = label[i];
    if (abs(predict - gt) < 1e-8)
      hit += 1;
    else
      miss += 1;
  }

  assert(hit + miss == input.size());

  return hit / static_cast<double>(hit + miss);
}

static void readPretrainedModel(string model_file, vector<double> &w,
                                double &b) {
  ifstream input_file;
  input_file.open(model_file);

  string line;
  while (getline(input_file, line)) {
    stringstream s(line);
    string number;
    uint32_t cnt = 0;

    while (getline(s, number, ',')) {
      if (cnt == 0) {
        b = stod(number);
        cnt += 1;
        continue;
      }

      w[cnt - 1] = stod(number);
      cnt += 1;
    }
  }

  input_file.close();
}

static void readTxtFile(vector<double> &input, const string &file_name,
                        uint32_t samples, const vector<uint32_t> &idxes = {}) {
  ifstream input_file;
  input_file.open(file_name);

  if (!input_file) {
    cout << file_name << " opens failed!" << endl;
    return;
  }

  cout << "Reading from file " << file_name << endl;
  double number;
  uint32_t cnt = 0;
  while (input_file >> number) {
    cnt += 1;
    if (idxes.size() == 0 ||
        find(idxes.begin(), idxes.end(), cnt) != idxes.end())
      input.push_back(number);

    if (input.size() == samples)
      break;
  }
  cout << "Read " << input.size() << " samples." << endl;

  input_file.close();
}

static void readCsvFile(vector<vector<double>> &dataset,
                        const string &file_name) {
  ifstream input_file;
  input_file.open(file_name);

  if (!input_file) {
    cout << file_name << " opens failed!" << endl;
    return;
  }
  cout << "Reading from file " << file_name << endl;

  vector<double> sample;
  string line;
  while (getline(input_file, line)) {
    if (line.find("feature") != string::npos)
      continue;

    sample.clear();

    stringstream s(line);
    string number;
    while (getline(s, number, ',')) {
      sample.push_back(stod(number));
    }

    dataset.push_back(sample);
  }

  input_file.close();
}

static void generateBatch(const vector<vector<double>> &dataset,
                          uint32_t &start_idx, uint32_t batch_size,
                          vector<vector<double>> &feature_batch,
                          vector<double> &label_batch) {
  feature_batch.clear();
  label_batch.clear();

  for (size_t i = 0; i < batch_size; i++) {
    uint32_t idx = (start_idx + i) % dataset.size();
    vector<double> sample = dataset[idx];

    label_batch.push_back(sample.back());
    sample.pop_back();
    feature_batch.push_back(sample);
  }

  start_idx = (start_idx + batch_size) % dataset.size();
}

static void printExpectedResult(uint32_t client_num,
                                const vector<double> &input, size_t slot_cnt) {
  // This function works when input from all clients are same.

  cout << "    + Expected result (" << client_num << " clients):" << endl;
  vector<double> true_result;
  // auto m = max_element(input.begin(), input.end(), absCompare);
  // double max_value = *m;

  for (auto &x : input) {
    true_result.push_back(computeLogarithm3rdDegree(x * client_num, 0));

    if (true_result.size() % slot_cnt == 0) {
      print_vector(true_result, 3, 7);
      true_result.clear();
    }
  }
  print_vector(true_result, 3, 7);
}

/*
Helper function: Prints the name of the example in a fancy banner.
*/
inline void print_example_banner(std::string title) {
  if (!title.empty()) {
    std::size_t title_length = title.length();
    std::size_t banner_length = title_length + 2 * 10;
    std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
    std::string banner_middle =
        "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";

    std::cout << std::endl
              << banner_top << std::endl
              << banner_middle << std::endl
              << banner_top << std::endl;
  }
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const seal::SEALContext &context) {
  auto &context_data = *context.key_context_data();

  /*
  Which scheme are we using?
  */
  std::string scheme_name;
  switch (context_data.parms().scheme()) {
  case seal::scheme_type::bfv:
    scheme_name = "BFV";
    break;
  case seal::scheme_type::ckks:
    scheme_name = "CKKS";
    break;
  default:
    throw std::invalid_argument("unsupported scheme");
  }
  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters :" << std::endl;
  std::cout << "|   scheme: " << scheme_name << std::endl;
  std::cout << "|   poly_modulus_degree: "
            << context_data.parms().poly_modulus_degree() << std::endl;

  /*
  Print the size of the true (product) coefficient modulus.
  */
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_modulus_size = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_modulus_size - 1; i++) {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;

  /*
  For the BFV scheme print the plain_modulus parameter.
  */
  if (context_data.parms().scheme() == seal::scheme_type::bfv) {
    std::cout << "|   plain_modulus: "
              << context_data.parms().plain_modulus().value() << std::endl;
  }

  std::cout << "\\" << std::endl;
}

/*
Helper function: Prints the `parms_id' to std::ostream.
*/
inline std::ostream &operator<<(std::ostream &stream,
                                seal::parms_id_type parms_id) {
  /*
  Save the formatting information for std::cout.
  */
  std::ios old_fmt(nullptr);
  old_fmt.copyfmt(std::cout);

  stream << std::hex << std::setfill('0') << std::setw(16) << parms_id[0] << " "
         << std::setw(16) << parms_id[1] << " " << std::setw(16) << parms_id[2]
         << " " << std::setw(16) << parms_id[3] << " ";

  /*
  Restore the old std::cout formatting.
  */
  std::cout.copyfmt(old_fmt);

  return stream;
}

/*
Helper function: Prints a matrix of values.
*/
template <typename T>
inline void print_matrix(std::vector<T> matrix, std::size_t row_size) {
  /*
  We're not going to print every column of the matrix (there are 2048). Instead
  print this many slots from beginning and end of the matrix.
  */
  std::size_t print_size = 5;

  std::cout << std::endl;
  std::cout << "    [";
  for (std::size_t i = 0; i < print_size; i++) {
    std::cout << std::setw(3) << std::right << matrix[i] << ",";
  }
  std::cout << std::setw(3) << " ...,";
  for (std::size_t i = row_size - print_size; i < row_size; i++) {
    std::cout << std::setw(3) << matrix[i]
              << ((i != row_size - 1) ? "," : " ]\n");
  }
  std::cout << "    [";
  for (std::size_t i = row_size; i < row_size + print_size; i++) {
    std::cout << std::setw(3) << matrix[i] << ",";
  }
  std::cout << std::setw(3) << " ...,";
  for (std::size_t i = 2 * row_size - print_size; i < 2 * row_size; i++) {
    std::cout << std::setw(3) << matrix[i]
              << ((i != 2 * row_size - 1) ? "," : " ]\n");
  }
  std::cout << std::endl;
}

/*
Helper function: Print line number.
*/
inline void print_line(int line_number) {
  std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

/*
Helper function: Convert a value into a hexadecimal string, e.g., uint64_t(17)
--> "11".
*/
inline std::string uint64_to_hex_string(std::uint64_t value) {
  return seal::util::uint_to_hex_string(&value, std::size_t(1));
}
