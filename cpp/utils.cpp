#include "utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#include <unistd.h>

char* get_cmd_option(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return nullptr;
}

bool cmd_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

float get_cmd_option_float(char **begin, char **end, const std::string &option, float _default) {
    char *result = get_cmd_option(begin, end, option);
    return result ? static_cast<float>(std::atof(result)) : _default;
}

int get_cmd_option_int(char **begin, char **end, const std::string &option, int _default) {
    char *result = get_cmd_option(begin, end, option);
    return result ? std::atoi(result) : _default;
}

std::string get_cmd_option_string(char **begin, char **end, const std::string &option, const std::string &_default) {
    char *result = get_cmd_option(begin, end, option);
    return result ? std::string(result) : _default;
}

std::string init_out_dir(int argc, char **argv) {
    std::string outdir = get_cmd_option_string(argv, argv + argc, "--outdir", "");
    if (outdir.empty()) {
        std::string tag = get_cmd_option_string(argv, argv + argc, "--tag", "");
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::string fmt = "%y-%m-%d_%H-%M-%S";
        std::ostringstream out_path;
        out_path << "output/" << tag << (!tag.empty() ? "_" : "") << std::put_time(std::localtime(&in_time_t), fmt.c_str());
        outdir = out_path.str();
    }

    std::cout << "Using " << outdir << " as output directory." << std::endl;
    system(("mkdir -p " + outdir).c_str());

    std::ofstream out_info(outdir + "/info.txt");
    for (int i= 0; i < argc; i++) {
        out_info << argv[i] << " ";
    }
    out_info << std::endl;
    out_info.close();

    return outdir;
}

int goto_out_dir(const std::string &out_dir) {
    return chdir(out_dir.c_str());
}

void ensure(bool condition, const std::string &error_message) {
    if (!condition) {
        throw std::runtime_error(error_message);
    }
}

float sqr(float x) {
    return x * x;
}

vec<std::pair<int, int>> make_pair_list(int n, bool top_right_triangle, bool no_diagonal) {
    vec<std::pair<int, int>> result;
    for (int i = 0; i < n; i++) {
        for (int j = (top_right_triangle ? i : 0); j < n; j++) {
            if (no_diagonal && i == j) {
                continue;
            }
            result.push_back(std::make_pair(i, j));
        }
    }
    return result;
}


