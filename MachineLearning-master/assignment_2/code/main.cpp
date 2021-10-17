#include <iostream>
#include <cstdlib>
#include "FileReader.h"
#include "Trainer.h"
#include "Tester.h"

using namespace std;


int main(int argc, char *argv[]) {
    vector<string> stopwords;
    if (argc == 5)
        stopwords = get_stopwords(argv[4]);
    FileReader trainf(argv[1]);
    FileReader testf(argv[2]);

    if (strcmp(argv[3], "lr") == 0) {
        LogisticRegressionTrainer train(trainf, stopwords);
        LogisticRegressionTester test(testf, train, stopwords);
    } else if (strcmp(argv[3], "nb") == 0) {
        NaiveBayesTrainer train(trainf, stopwords);
        NaiveBayesTester test(testf, train, stopwords);
    } else
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}