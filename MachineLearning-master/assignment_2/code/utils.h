#include <set>
#include <string>
#include <sstream>

using namespace std;

bool check_if_stopword(const string &word, vector<string> stopwords) {
    if (find(stopwords.begin(), stopwords.end(), word) != stopwords.end())
        return false;
    return true;
}

static string remove_special_symbols(const string &str) {
    string modified_str;
    for (char i: str) {
        if (isalpha(i))
            modified_str += i;
    }
    return modified_str;
}

static set<string> get_tokens(const string &str, const vector<string> &stopwords) {
    istringstream ss(str);
    set<string> tokens;
    do {
        string word;
        ss >> word;
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        word = remove_special_symbols(word);
        if (check_if_stopword(word, stopwords))
            tokens.insert(word);
    } while (ss);
    return tokens;
}

static vector<string> get_stopwords(const string &path) {
    ifstream in(path);
    vector<string> vecOfStrs;
    string token;
    while (getline(in, token))
        if (!token.empty())
            vecOfStrs.push_back(token);
    return vecOfStrs;
}