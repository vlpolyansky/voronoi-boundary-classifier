#include "utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

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

#ifdef _OPENMP
void omp_optimize_num_threads(int niter) {
    int num_proc = omp_get_num_procs();
    int min_blocks = (niter + num_proc - 1) / num_proc;
    int min_num_threads = (niter + min_blocks - 1) / min_blocks;
    min_num_threads = std::min(num_proc, min_num_threads);
    std::cout << "Using " << min_num_threads << " threads" << std::endl;
    omp_set_num_threads(min_num_threads);
}
#endif

big_float pow(big_float a, int b) {
    big_float res = 1;
    while (b > 0) {
        if (b & 1) {
            res *= a;
            b--;
        } else {
            a = a * a;
            b >>= 1;
        }
    }
    return res;
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


void simple_progress(long long i, long long niter, int multiplier) {
    if (i == 0 || i * multiplier / niter > (i - 1) * multiplier / niter) {
        printf("\r%2.2f%%", i * 100.0 / niter);
        fflush(stdout);
//        std::cout << "\r" << (i * 100.0 / niter) << "%";
//        std::cout.flush();
    }
    if (i + 1 == niter) {
        printf("\r     %%\r");
        fflush(stdout);
    }
}

bool _AWAITING_NEW_PROGRESS = false;

void activate_next_progress() {
    _AWAITING_NEW_PROGRESS = true;
}

void deactivate_next_progress() {
    _AWAITING_NEW_PROGRESS = false;
}

bool reserve_progress_if_available() {
    bool ret = false;
#ifdef _OPENMP
#pragma omp master
#endif
    {
        ret = _AWAITING_NEW_PROGRESS;
        if (_AWAITING_NEW_PROGRESS) {
            _AWAITING_NEW_PROGRESS = false;
        }
    };

    return ret;
}
