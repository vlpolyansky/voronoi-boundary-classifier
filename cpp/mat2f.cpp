#include "mat2f.h"

#include <fstream>
#include <iostream>

#include "utils.h"

long long mat2f::MAX_CHUNK_SIZE = 100000000;

mat2f::mat2f() : mat2f(0, 0, .0f, true) {
}

mat2f::mat2f(int rows, int cols, bool one_piece) : mat2f(rows, cols, .0f, one_piece) {}

mat2f::mat2f(int rows, int cols, float value, bool one_piece) : rows(rows), cols(cols), one_piece(one_piece) {
    int temp;
    if (one_piece) {
        temp = rows * 2 - 1;
    } else {
        temp = int(MAX_CHUNK_SIZE / cols);
    }

    finish_initialization(rows, cols, value, temp);
}

mat2f::mat2f(int rows, int cols, int chunk_rows) : mat2f(rows, cols, .0f, chunk_rows) {}

mat2f::mat2f(int rows, int cols, float value, int chunk_rows) : rows(rows), cols(cols) {
    ensure(chunk_rows > 0,
           "Bad chunk_rows: " + std::to_string(chunk_rows) + ". Did you mean to use a float value?");
    ensure((chunk_rows & (chunk_rows - 1)) == 0,
           "Bad chunk_rows: " + std::to_string(chunk_rows) + ". Need to be a power of two.");

    finish_initialization(rows, cols, value, chunk_rows);
}

void mat2f::finish_initialization(int rows, int cols, float value, int prelim_chunk_rows) {
    int temp = prelim_chunk_rows;
    normal_chunk_rows_bit = 0;
    while (temp > 1) {
        temp >>= 1;
        normal_chunk_rows_bit++;
    }

    normal_chunk_rows = 1 << normal_chunk_rows_bit;
    chunks = (rows + normal_chunk_rows - 1) / normal_chunk_rows;
    last_chunk_rows = rows - normal_chunk_rows * (chunks - 1);

    data = vec2f(chunks);
    for (int i = 0; i < chunks; i++) {
        data[i] = vec1f((i + 1 < chunks ? normal_chunk_rows : last_chunk_rows) * cols, value);
    }
}

void mat2f::fill_from(const range<float> &range) {
    ensure(range.size() == rows * cols, "Source vector has different size from mat2f size.");

    for (int chunk_id = 0; chunk_id < chunks; chunk_id++) {
        data[chunk_id] = vec1f(range.begin + chunk_id * normal_chunk_rows * cols,
                               range.begin + chunk_id * normal_chunk_rows * cols + get_chunk_rows(chunk_id) * cols);
    }
}

int mat2f::fill_submat_from(const range<float> &range, int offset) {
    ensure(range.size() % cols == 0, "Inserted data should contain a whole number of rows.");
    int num = range.size() / cols;
    ensure(offset + num <= rows, "Data does not fit into the matrix.");
    for (int i = 0; i < num; i++, offset++) {
        std::copy(range.begin + i * cols, range.begin + (i + 1) * cols, &operator()(offset, 0));
    }
    return offset;
}

float& mat2f::operator()(int row, int col) {
    return data[row >> normal_chunk_rows_bit][(row & (normal_chunk_rows - 1)) * cols + col];
}

float mat2f::operator()(int row, int col) const {
    return data[row >> normal_chunk_rows_bit][(row & (normal_chunk_rows - 1)) * cols + col];
}

range<float> mat2f::operator[](int row) {
    int chunk_id = row >> normal_chunk_rows_bit;
    int row_id = (row & (normal_chunk_rows - 1)) * cols;
    return {data[chunk_id].begin() + row_id, data[chunk_id].begin() + (row_id + cols)};
}

int mat2f::get_chunks_num() const {
    return chunks;
}

vec1f& mat2f::get_data(int chunk_id) {
    if (chunk_id == -1) {
        ensure(chunks == 1, "To get the full data, the number of chunks should be equal to one.");
        return data[0];
    }
    return data[chunk_id];
}

int mat2f::get_chunk_rows(int chunk_id) const {
    return chunk_id + 1 < chunks ? normal_chunk_rows : last_chunk_rows;
}

mat2f load_mat2d(const std::string &filename, int n, int m, bool one_piece) {
    std::ifstream in(filename);
    if (n == -1 || m == -1) {
        // header
        in >> n >> m;
    }
    mat2f mat(n, m, 0, one_piece);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            in >> mat(i, j);
        }
    }

    return mat;
}

void save_mat2d(const mat2f &mat, const std::string &filename, bool header) {
    std::ofstream out(filename);
    if (header) {
        out << mat.rows << " " << mat.cols << std::endl;
    }
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            out << mat(i, j) << " ";
        }
        out << "\n";
    }
    out.close();
}

int mat2f::predict_chunk_rows(int cols) {
    int temp = int(MAX_CHUNK_SIZE / cols);
    int bit = 0;
    while (temp > 1) {
        temp >>= 1;
        bit++;
    }
    return 1 << bit;
}

void mat2f::reset(float val) {
    for (int i = 0; i < chunks; i++) {
        std::fill(data[i].begin(), data[i].end(), val);
    }
}
