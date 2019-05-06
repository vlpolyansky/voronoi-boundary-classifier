#pragma once

#include "geometry.h"
#include "range.h"

#include <string>

class mat2f {
public:
    mat2f();
    mat2f(int rows, int cols, bool one_piece = true);
    mat2f(int rows, int cols, float value, bool one_piece = true);
    mat2f(int rows, int cols, int chunk_rows);
    mat2f(int rows, int cols, float value, int chunk_rows);

    void fill_from(const range<float> &range);
    int fill_submat_from(const range<float> &range, int offset);

    float& operator() (int row, int col);
    float  operator() (int row, int col) const;

    range<float> operator[] (int row);

    int get_chunks_num() const;
    vec1f& get_data(int chunk_id = -1);
    int get_chunk_rows(int chunk_id = 0) const;

    void reset(float val = 0);

private:
    void finish_initialization(int rows, int cols, float value, int prelim_chunk_rows);

public:
    vec2f data; // chunks by rows
    int rows, cols;

private:
    bool one_piece;
    int chunks;
    int normal_chunk_rows, normal_chunk_rows_bit, last_chunk_rows;

public:
    static int predict_chunk_rows(int cols);
    static long long MAX_CHUNK_SIZE; // = 100000000;
};



mat2f load_mat2d(const std::string &filename, int n = -1, int m = -1, bool one_piece = true);

void save_mat2d(const mat2f &mat, const std::string &filename, bool header = false);
