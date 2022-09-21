#include "logistic_regression.h"
#include "kms.h"

using namespace std;

int main(int argc, char* argv[]) {
    LogisticRegression lr;
    LHE_KMS kms;
    kms.generate(COMPUTE_MODE::Train, 16384, sec_level_type::tc128);
    if (argc <= 2) {
        cout << "Please input train and eval data path." << endl;
        exit(1);
    }
    vector<vector<double>> training_set;
    vector<vector<double>> eval_set;
    cout << string(argv[1]) << endl;
    cout << string(argv[2]) << endl;

    if (argc >2) {
        lr.set_batch_size(atoi(argv[3]));
    }

    if (argc >3) {
        lr.set_max_iter(atoi(argv[4]));
    }

    if (argc > 4) {
        lr.set_learning_rate(atof(argv[5]));
    }

    vector<string> secrets;
    secrets.push_back(kms.getEncryptionParamters().str());
    secrets.push_back(kms.getPublicKey().str());
    secrets.push_back(kms.getRelinearKey().str());
    secrets.push_back(kms.getSecretKey().str());
    // cout << secrets[0] << endl;
    lr.init(secrets);
    cout << "Init LR model" << endl;
    readCsvFile(training_set, string(argv[1]));
    readCsvFile(eval_set, string(argv[2]));
    cout << "Read dataset" << endl;
    lr.set_learning_rate(0.001);
    lr.set_data(training_set, eval_set);
    cout << "Set dataset" << endl;
    lr.train();
    cout << "Train complete" << endl;
    return 0;
}
