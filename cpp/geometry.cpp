#include "geometry.h"

#include "fstream"

vec2f make_vec2d(int n, int m, float val) {
    return vec2f(n, vec1f(m, val));
}

vec3f make_vec3d(int n, int m, int k, float val) {
    return vec3f(n, make_vec2d(m, k, val));
}

vec1f operator+(const range_d &a, const range_d &b) {
    assert(a.size() == b.size());
    vec1f c(a.size());
    for (int i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

vec1f operator-(const range<float> &a, const range<float> &b) {
    assert(a.size() == b.size());
    vec1f c(a.size());
    for (int i = 0; i < a.size(); i++) {
        c[i] = a[i] - b[i];
    }
    return c;
}

vec1f operator*(const range<float> &a, float k) {
    vec1f c(a.size());
    for (int i = 0; i < a.size(); i++) {
        c[i] = a[i] * k;
    }
    return c;
}

vec1f operator*(float k, const range<float> &a) {
    return a * k;
}

vec1f operator/(const range<float> &a, float k) {
    return a * (1. / k);
}

float cross_2d(const range<float> &a, const range<float> &b) {
    assert(a.size() == 2);
    assert(a.size() == 2);
    return a[0] * b[1] - a[1] * b[0];
}


float operator*(const range<float> &a, const range<float> &b) {
    assert(a.size() == b.size());
    float c = 0;
    for (int i = 0; i < a.size(); i++) {
        c += a[i] * b[i];
    }
    return c;
}

float length_sqr(range<float> a) {
    return a * a;
}

float length(const range<float> &a) {
    return std::sqrt(length_sqr(a));
}

float dist_sqr(const range<float> &a, const range<float> &b) {
    return length_sqr(a - b);
}


void save_matrix(const vec2f &mat, const std::string &filename, bool header) {
    std::ofstream out(filename);
    if (header) {
        out << mat.size() << " " << mat[0].size() << std::endl;
    }
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            out << mat[i][j] << " ";
        }
        out << std::endl;
    }
    out.close();
}

vec2f load_matrix(const std::string &filename, int n, int m) {
    std::ifstream in(filename);
    if (n == -1 || m == -1) {
        // header
        in >> n >> m;
    }
    vec2f mat = make_vec2d(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            in >> mat[i][j];
        }
    }

    return mat;
}

void save_vector(const range<int> &vec, const std::string &filename, bool header) {
    std::ofstream out(filename);
    if (header) {
        out << vec.size() << std::endl;
    }
    for (int i = 0; i < vec.size(); i++) {
        out << vec[i] << " ";
    }
    out << "\n";
    out.close();
}


vec1i load_vec1i(const std::string &filename, int n) {
    std::ifstream in(filename);
    if (n == -1) {
        // header
        in >> n;
    }
    vec1i vec(n);
    for (int i = 0; i < n; i++) {
        in >> vec[i];
    }

    return vec;
}