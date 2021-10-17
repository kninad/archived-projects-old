#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace std;

class Tester {
public:
    double acc, precision, recall, f_score;

    void print_stats() {
        cout << "Accuracy = " << acc << "\n";
        cout << "Precision = " << precision << "\n";
        cout << "Recall = " << recall << "\n";
        cout << "F-score = " << f_score << "\n";
    }

    void log_stats(pair<int, int> ham_preds, pair<int, int> spam_preds) {
        int correct_count = ham_preds.first + spam_preds.first;
        int total_count = ham_preds.second + spam_preds.second;
        acc = (double) correct_count / (double) total_count;
        precision = (double) ham_preds.first / (double) (ham_preds.first + spam_preds.second - spam_preds.first);
        recall = (double) ham_preds.first / (double) (ham_preds.first + ham_preds.second - ham_preds.first);
        f_score = (2 * precision * recall) / (precision + recall);
    }
};

class NaiveBayesTester : Tester {
public:
    explicit NaiveBayesTester(FileReader f, const NaiveBayesTrainer &t, const vector<string> &stopwords) {
        map<string, string> x = f.get_directory_names();
        vector<string> ham_docs = f.get_file_names(x[ham]);
        vector<string> spam_docs = f.get_file_names(x[spam]);
        pair<int, int> ham_counts = get_predictions(t, ham_docs, ham, stopwords);
        pair<int, int> spam_counts = get_predictions(t, spam_docs, spam, stopwords);
        log_stats(ham_counts, spam_counts);
        print_stats();
    }

private:
    pair<int, int>
    get_predictions(NaiveBayesTrainer t, vector<string> docs, const string &type, const vector<string> &stopwords) {
        double log_ham_prior, log_spam_prior;
        int correct_count = 0, total_count = 0;
        vector<string>::iterator it;
        set<string>::iterator itt;
        for (it = docs.begin(); it != docs.end(); it++) {
            log_ham_prior = log(t.ham_prior);
            log_spam_prior = log(t.spam_prior);
            set<string> tokens = get_tokens(*it, stopwords);
            for (itt = tokens.begin(); itt != tokens.end(); itt++) {
                if (t.ham_counts[*itt] != 0)
                    log_ham_prior += log(t.ham_counts[*itt]);
                if (t.spam_counts[*itt] != 0)
                    log_spam_prior += log(t.spam_counts[*itt]);
            }
            if (type == ham)
                if (log_ham_prior < log_spam_prior)
                    correct_count++;
            if (type == spam)
                if (log_spam_prior < log_ham_prior)
                    correct_count++;
            total_count++;
        }
        return make_pair(correct_count, total_count);
    }
};

class LogisticRegressionTester : Tester {
public:

    explicit LogisticRegressionTester(FileReader f, const LogisticRegressionTrainer &t,
                                      const vector<string> &stopwords) {
        map<string, string> x = f.get_directory_names();
        vector<string> ham_docs = f.get_file_names(x[ham]);
        vector<string> spam_docs = f.get_file_names(x[spam]);
        vector<map<string, double> > ham_counts = get_token_counts(ham_docs, stopwords);
        vector<map<string, double> > spam_counts = get_token_counts(spam_docs, stopwords);
        pair<int, int> ham_predictions = get_predictions(t, ham_counts, ham);
        pair<int, int> spam_predictions = get_predictions(t, spam_counts, spam);
        log_stats(ham_predictions, spam_predictions);
        print_stats();
    }

private:
    pair<int, int>
    get_predictions(LogisticRegressionTrainer t, const vector<map<string, double> > &counts, const string &type) {
        double sum = t.bias;
        int correct_count = 0, total_count = 0;
        for (auto &item: counts) {
            for (auto &map_item: item)
                sum += map_item.second * t.weights[map_item.first];
            if (type == ham)
                if (sum >= 0)
                    correct_count++;
            if (type == spam)
                if (sum < 0)
                    correct_count++;
            total_count++;
        }
        return make_pair(correct_count, total_count);
    }

    vector<map<string, double> > get_token_counts(vector<string> texts, const vector<string> &stopwords) {
        vector<map<string, double> > counts;
        vector<string>::iterator it;
        set<string>::iterator itt;
        set<string> tokens;
        map<string, double> token_counts;
        for (it = texts.begin(); it != texts.end(); ++it) {
            tokens = get_tokens(*it, stopwords);
            for (itt = tokens.begin(); itt != tokens.end(); itt++)
                token_counts[*itt]++;
            counts.push_back(token_counts);
            token_counts.clear();
        }
        return counts;
    }
};