#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>

#include "utils.h"

using namespace std;

class NaiveBayesTrainer {
public:
    double ham_prior, spam_prior;
    map<string, double> ham_counts, spam_counts;

    explicit NaiveBayesTrainer(FileReader f, const vector<string> &stopwords) {
        map<string, string> x = f.get_directory_names();
        vector<string> ham_docs = f.get_file_names(x[ham]);
        vector<string> spam_docs = f.get_file_names(x[spam]);
        ham_counts = get_token_counts(ham_docs, stopwords);
        spam_counts = get_token_counts(spam_docs, stopwords);
        ham_prior = (double) ham_docs.size() / (double) (ham_docs.size() + spam_docs.size());
        spam_prior = (double) spam_docs.size() / (double) (ham_docs.size() + spam_docs.size());
        ham_counts = compute_conditional(ham_counts);
        spam_counts = compute_conditional(spam_counts);
    }

private:
    static map<string, double> get_token_counts(vector<string> texts, const vector<string> &stopwords) {
        map<string, double> counts;
        vector<string>::iterator it;
        set<string>::iterator itt;
        set<string> tokens;
        for (it = texts.begin(); it != texts.end(); ++it) {
            tokens = get_tokens(*it, stopwords);
            for (itt = tokens.begin(); itt != tokens.end(); itt++)
                counts[*itt]++;
        }
        return counts;
    }

    static map<string, double> compute_conditional(map<string, double> counts) {
        double length = 0.0;
        map<string, double>::iterator it;
        for (it = counts.begin(); it != counts.end(); it++)
            length += it->second + 1;
        for (it = counts.begin(); it != counts.end(); it++)
            it->second = (it->second + 1) / length;
        return counts;
    }
};

class LogisticRegressionTrainer {
public:
    int num_iterations = 50;
    double eta = 1e-5, bias = 1.0;
    map<string, double> weights, g;
    map<map<string, double>, double> p;

    explicit LogisticRegressionTrainer(FileReader f, const vector<string> &stopwords) {
        map<string, string> x = f.get_directory_names();
        vector<string> ham_docs = f.get_file_names(x[ham]);
        vector<string> spam_docs = f.get_file_names(x[spam]);
        vector<map<string, double> > ham_counts = get_token_counts(ham_docs, stopwords);
        vector<map<string, double> > spam_counts = get_token_counts(spam_docs, stopwords);
        trainer(ham_counts, spam_counts);
    }

    vector<map<string, double> > get_token_counts(vector<string> texts, const vector<string> &stopwords) {
        vector<map<string, double> > counts;
        vector<string>::iterator it;
        set<string>::iterator itt;
        set<string> tokens;
        map<string, double> token_counts;
        for (it = texts.begin(); it != texts.end(); ++it) {
            tokens = get_tokens(*it, stopwords);
            for (itt = tokens.begin(); itt != tokens.end(); itt++) {
                token_counts[*itt]++;
                weights[*itt] = 0.0; // weight matrix to be trained
            }
            counts.push_back(token_counts);
            token_counts.clear();
        }
        return counts;
    }

    void trainer(const vector<map<string, double> > &ham_counts, vector<map<string, double> > &spam_counts) {
        int i;
        while (num_iterations--) {
            train_one_cycle(ham_counts, 1);
            train_one_cycle(spam_counts, 0);
            for (auto &v:weights)
                weights[v.first] += eta * (g[v.first] - v.second);
        }
    }

    void train_one_cycle(const vector<map<string, double> > &counts, int label) {
        double sum;
        for (auto map_item:counts) {
            sum = bias;
            for (const auto &vocab:weights)
                sum += vocab.second * map_item[vocab.first];
            p[map_item] = 1.0 / (1.0 + pow(2, -sum));
        }
        for (const auto &vocab:weights)
            g[vocab.first] = 0;
        for (auto map_item:counts)
            for (const auto &vocab:weights)
                g[vocab.first] += map_item[vocab.first] * (label - p[map_item]);
    }

};