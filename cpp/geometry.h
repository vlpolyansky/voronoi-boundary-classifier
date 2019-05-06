#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <string>

#include "range.h"

using vec1i = std::vector<int>;
using vec1f = std::vector<float>;
using vec2f = std::vector<vec1f>;
using vec3f = std::vector<vec2f>;

using range_d = range<float>;

vec2f make_vec2d(int n, int m, float val = 0);

vec3f make_vec3d(int n, int m, int k, float val = 0);

vec1f operator+(const range_d &a, const range_d &b);

vec1f operator-(const range<float> &a, const range<float> &b);

vec1f operator*(const range<float> &a, float k);

vec1f operator*(float k, const range<float> &a);

vec1f operator/(const range<float> &a, float k);

float cross_2d(const range<float> &a, const range<float> &b);

float operator*(const range<float> &a, const range<float> &b);

float length_sqr(range<float> a);

float length(const range<float> &a);

float dist_sqr(const range<float> &a, const range<float> &b);


void save_matrix(const vec2f &mat, const std::string &filename, bool header = false);

vec2f load_matrix(const std::string &filename, int n = -1, int m = -1);

void save_vector(const range<int> &vec, const std::string &filename, bool header = false);

vec1i load_vec1i(const std::string &filename, int n = -1);