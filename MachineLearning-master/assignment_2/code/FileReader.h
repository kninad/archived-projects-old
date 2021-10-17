#include <string>
#include <map>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include "constants.h"

using namespace std;

class FileReader {
public:
    string dir_path;

    FileReader(string path) {
        dir_path = path;
    }

    map<string, string> get_directory_names() {
        DIR *dirp = opendir(dir_path.c_str());
        map<string, string> contents;
        struct dirent *dp;
        while ((dp = readdir(dirp)) != nullptr) {
            string sub_dir_path;
            *dir_path.end() == '/' ? sub_dir_path = dir_path + dp->d_name : sub_dir_path = dir_path + "/" + dp->d_name;
            if (sub_dir_path.find(ham) != string::npos)
                contents.insert(pair<string, string>(ham, sub_dir_path));
            if (sub_dir_path.find(spam) != string::npos)
                contents.insert(pair<string, string>(spam, sub_dir_path));
        }
        closedir(dirp);
        return contents;
    }

    vector<string> get_file_names(string path) {
        DIR *dirp = opendir(path.c_str());
        vector<string> contents;
        struct dirent *dp;
        while ((dp = readdir(dirp)) != nullptr) {
            string filename = path + "/" + dp->d_name;
            if (filename.substr(filename.size() - 4) == string(file_ext))
                contents.push_back(get_file_contents(filename));
        }
        closedir(dirp);
        return contents;
    }

    string get_file_contents(string path) {
        ifstream t(path);
        string str((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
        return str;
    }
};