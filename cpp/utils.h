#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <string>
#include <boost/multiprecision/cpp_dec_float.hpp>

#define PI 3.141592653589793238462643383279
#define PI_f 3.141592653589793238462643383279f

template<typename T>
using vec = std::vector<T>;

template<typename T>
using vec2D = vec<vec<T>>;

#ifndef BIG_FLOAT_PRECISION
#define BIG_FLOAT_PRECISION 10
#endif
using big_float = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<BIG_FLOAT_PRECISION>>;


char* get_cmd_option(char **begin, char **end, const std::string &option);

float get_cmd_option_float(char **begin, char **end, const std::string &option, float _default = 0.0);

int get_cmd_option_int(char **begin, char **end, const std::string &option, int _default = 0);

std::string get_cmd_option_string(char **begin, char **end, const std::string &option, const std::string &_default = "");

bool cmd_option_exists(char **begin, char **end, const std::string &option);

std::string init_out_dir(int argc, char **argv);
int goto_out_dir(const std::string &out_dir);

void ensure(bool condition, const std::string &error_message);

#ifdef _OPENMP
void omp_optimize_num_threads(int niter);
#endif

big_float pow(big_float a, int b);

float sqr(float x);

vec<std::pair<int, int>> make_pair_list(int n, bool top_right_triangle = false, bool no_diagonal = false);


// ---------------- Progress tracking ----------------
void simple_progress(long long i, long long niter, int multiplier = 100);
void activate_next_progress();
void deactivate_next_progress();
bool reserve_progress_if_available();