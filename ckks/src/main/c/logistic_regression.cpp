#include "logistic_regression.h"

string LogisticRegression::inference_chunk(const string &x, uint32_t chunk_id) {
  cout << "Inference chunk " << chunk_id << endl;

  chrono::high_resolution_clock::time_point time_start, time_end;
  chrono::microseconds time_diff;

  stringstream ss;

  Ciphertext x_encrypted;
  ss.str("");
  ss << x;
  x_encrypted.load(*context_, ss);

  string result{""};
  time_start = chrono::high_resolution_clock::now();
  // Ciphertext result_encrypted = computeLogarithm3rdDegree(x_encrypted,
  // 0);
  Ciphertext result_encrypted =
      ckks_common_->computeSigmoid7thDegree(x_encrypted);
  time_end = chrono::high_resolution_clock::now();

  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "Sigmoid took " << time_diff.count() << " microseconds." << endl;

  result_encrypted.save(ss);
  result = ss.str();
  return result;
}

void LogisticRegression::train() {
  double scale = pow(2.0, 40);
  size_t slot_cnt = encoder_->slot_count();

  uint32_t sample_num = training_set_.size();
  uint32_t feature_size = training_set_[0].size() - 1;

  uint32_t max_iter = max_iter_;
  uint32_t iter_idx = 0;
  uint32_t epoch_idx = 0;
  uint32_t batch_size = min(batch_size_, sample_num);
  uint32_t batch_num = floor(sample_num / batch_size);

  // Print LR configuration
  cout << "-------------------------------Setting--------------------------" << endl;
  cout << "Batch Size: " << batch_size << endl;
  cout << "Max Iteration: " << max_iter << endl;
  cout << "Learning Rate: " << learning_rate_ << endl;

  double tol = 1e-5;

  double prev_loss = 0.0;
  uint32_t sample_idx = 0;

  random_device rd;
  mt19937 mt(rd());

  vector<vector<double>> eval_features;
  vector<double> eval_labels;
  uint32_t eval_idx = 0;

  generateBatch(eval_set_, eval_idx, eval_set_.size(), eval_features,
                eval_labels);

  chrono::high_resolution_clock::time_point epoch_start, epoch_end;
  chrono::milliseconds epoch_diff;

  while (iter_idx < max_iter) {
    double acc;
    if (iter_idx % batch_num == 0) {
      epoch_end = chrono::high_resolution_clock::now();
      if (epoch_end > epoch_start) {
        epoch_diff = chrono::duration_cast<chrono::milliseconds>(epoch_end - epoch_start);
        cout << "Epoch time: " << epoch_diff.count() << " ms." << endl;
      }
      shuffle(training_set_.begin(), training_set_.end(), mt);
      cout << "-------------------------Epoch:";
      cout << setw(6) << epoch_idx << "---------------------------" << endl;
      epoch_idx++;
      sample_idx = 0;
    }

    vector<vector<double>> feature_batch;
    vector<double> label_batch;
    generateBatch(training_set_, sample_idx, batch_size, feature_batch,
                  label_batch);

    vector<double> forward_batch = local_forward(feature_batch);
    if (iter_idx % batch_num == 0) {
      acc = localValidate(forward_batch, label_batch);
      cout << "Training Accuracy: " << acc << endl;
      vector<double> eval_inputs = local_forward(eval_features);
      double accuracy = localValidate(eval_inputs, eval_labels);
      cout << "Eval Accuracy: " << accuracy << endl;
    }

    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_diff;

    time_start = chrono::high_resolution_clock::now();
    vector<string> input_chunks = preprocess(forward_batch);
    vector<string> target_chunks = preprocess(label_batch);
    time_end = chrono::high_resolution_clock::now();

    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    // cout << "Preprocess took " << time_diff.count() << " microseconds." <<
    // endl;

    // cout << "Evaluating BCE loss computation..." << endl;
    vector<TrainFeedbacks> feedbacks;
    for (uint32_t i = 0; i < input_chunks.size(); i++) {
      feedbacks.push_back(train_chunk(input_chunks[i], target_chunks[i], i));
    }

    time_start = chrono::high_resolution_clock::now();
    // printExpectedLosses(1, forward_batch, label_batch, slot_cnt);
    time_end = chrono::high_resolution_clock::now();
    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    // cout << "Local computation took " << time_diff.count() << "
    // microseconds." << endl;

    // cout << "    + Computed result:" << endl;
    vector<double> loss_batch;
    vector<double> backward_batch;
    uint32_t remains = batch_size;

    time_start = chrono::high_resolution_clock::now();
    for (auto fb : feedbacks) {
      stringstream ss;
      ss << fb.loss;

      Ciphertext loss_encrypted;
      loss_encrypted.load(*context_, ss);

      vector<double> loss_plain;
      loss_plain = encryptor_->decrypt(loss_encrypted);

      ss.str("");
      ss << fb.backwards;

      Ciphertext backwards_encrypted;
      backwards_encrypted.load(*context_, ss);

      vector<double> backwards_plain;
      backwards_plain = encryptor_->decrypt(backwards_encrypted);

      uint32_t actual_size = min(remains, (uint32_t)loss_plain.size());
      remains -= actual_size;

      loss_plain.resize(actual_size);
      // cout << "Loss: " << endl;
      // print_vector(loss_plain, 3, 7);
      loss_batch.insert(loss_batch.end(), loss_plain.begin(), loss_plain.end());

      backwards_plain.resize(actual_size);
      // cout << "Backwards: " << endl;
      // print_vector(backwards_plain, 3, 7);
      backward_batch.insert(backward_batch.end(), backwards_plain.begin(),
                            backwards_plain.end());
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    // cout << "Post-process took " << time_diff.count() << " microseconds." <<
    // endl;

    // ofstream output_file;
    // output_file.open("../../temp/loss_he_" + to_string(estimator_id_) +
    // ".txt", ios::out | ios::trunc); ostream_iterator<double>
    // output_iter(output_file, "\n"); copy(loss_batch.begin(),
    // loss_batch.end(), output_iter); output_file.close();

    // double decay = pow(0.1, static_cast<double>(iter_idx / 100));
    weight_update(backward_batch, feature_batch, iter_idx);

    // check converge
    auto abs_val = [](auto val, auto sum) { return sum + std::fabs(val); };
    double curr_loss =
        accumulate(loss_batch.begin(), loss_batch.end(), 0.0, abs_val) /
        loss_batch.size();
    // cout << "Iter: " << iter_idx << ", loss: " << curr_loss << endl;
    if (iter_idx % batch_num == 0) {
      cout << "Loss: " << curr_loss << endl;
      epoch_start = chrono::high_resolution_clock::now();
    }
    iter_idx += 1;

    // if (fabs(curr_loss - prev_loss) < tol) {
    //   cout << "Training converged with loss " << curr_loss << endl;
    //   break;
    // }

    prev_loss = curr_loss;
  }

  cout << "Training finished!" << endl;
  cout << "----------------------------------------------------------------" << endl;
}

vector<string> LogisticRegression::preprocess(vector<double> input) {
  // normalization
  // use batch norm for now
  // auto m = max_element(input.begin(), input.end(), absCompare);
  // double max_value = *m;
  // cout << "Max value is " << max_value << endl;
  // for_each(input.begin(), input.end(), [max_value](double &i)
  //          { i /= max_value; });
  // cout << "After normalization:" << endl;
  // print_vector(input, 3, 7);

  // slicing
  size_t slot_cnt = encoder_->slot_count();
  size_t chunk_cnt = ceil(input.size() / static_cast<double>(slot_cnt));
  vector<vector<double>> chunks(chunk_cnt, vector<double>(slot_cnt, 0));

  for (auto idx = 0; idx < input.size(); idx++) {
    unsigned chunk_idx = floor(idx / static_cast<double>(slot_cnt));
    unsigned element_idx = idx % slot_cnt;
    chunks[chunk_idx][element_idx] = input[idx];
  }

  // encryption
  // cout << "After slicing:" << endl;

  vector<string> encrypted_slices;
  for (auto v : chunks) {
    // print_vector(v, 3, 7);
    stringstream ss = encryptor_->encrypt(v);
    encrypted_slices.push_back(ss.str());
  }

  return encrypted_slices;
}

TrainFeedbacks LogisticRegression::train_chunk(const string &x, const string &y,
                                               uint32_t chunk_id) {
  stringstream ss;

  Ciphertext x_encrypted;
  ss.str("");
  ss << x;
  x_encrypted.load(*context_, ss);

  Ciphertext y_encrypted;
  ss.str("");
  ss << y;
  y_encrypted.load(*context_, ss);

  chrono::high_resolution_clock::time_point time_start, time_end;
  chrono::microseconds time_diff;

  time_start = chrono::high_resolution_clock::now();
  Ciphertext pred_encrypted =
      ckks_common_->computeSigmoid3rdDegree(x_encrypted);
  // Ciphertext pred_encrypted = computeSigmoid7thDegree(x_encrypted);
  Ciphertext loss_encrypted =
      ckks_common_->computeBCE(pred_encrypted, y_encrypted);
  Ciphertext backwards_encrypted =
      ckks_common_->computeBackwards(pred_encrypted, y_encrypted);
  time_end = chrono::high_resolution_clock::now();

  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  // cout << "Loss and backwards computation took " << time_diff.count()
  //      << " microseconds." << endl;

  TrainFeedbacks feedback;
  ss.str("");
  loss_encrypted.save(ss);
  feedback.loss = ss.str();

  ss.str("");
  backwards_encrypted.save(ss);
  feedback.backwards = ss.str();

  return feedback;
}
