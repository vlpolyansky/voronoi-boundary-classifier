#pragma once

#include <string>
#include <random>
#include <chrono>
#include <functional>
#include <memory>

#include <libutils/misc.h>

#include "geometry.h"
#include "weights.h"
#include "mat2f.h"
#include "KDTree.h"

class VoronoiClassifier {
public:
    explicit VoronoiClassifier(int seed = 239);

    void load_train_data(const std::string &filename, int max_n = -1);
    void load_test_data(const std::string &filename);

    /* Implementation specific method, doesn't include setting of a weight */
    void prepare(int argc, char **argv);

    void init_kernels();

    void set_weight(const std::shared_ptr<Weight> &weight);

    /* Important to run one of the following two functions */
    void precalculate_dxdx();
    void load_dxdx(const std::string &filename);

    void save_dxdx(const std::string &filename);

    /* Use train data as test data as well (each test point tested against all other points) */
    void init_selftest();

    void perform_iterations_and_update(int niter);
    void update_statistics();

    void load_classification_data(const std::string &directory);
    void save_classification_data(const std::string &directory);

    float series_to_integral_value(int i, int k_);

private:
    std:: mt19937 random_engine;
    std::normal_distribution<float> normal_distribution;
    std::uniform_real_distribution<float> uniform_real_distribution;
    bool has_next_rand_normal = false;
    float next_rand_normal;
    unsigned long long rand_state;
    float rand_normal();
    float rand_float();
    vec1f rand_vec_on_sphere(int ndim);
    float sphere_area(int ndim);

    void cast_ray(int test_idx, const range<float> &u);
    int nn(const range<float> &x, float *distance, float margin_sqr = -1);

public:
    int train_n, test_n, d, k;
    mat2f train_data, test_data;
    vec1i train_labels, test_labels;
    std::shared_ptr<Weight> weight;

private:
    int test_max_chunk_size;
    int train_max_chunk_size;

    std::shared_ptr<KDTree> kd_tree;

public:
    mat2f series;
    vec1i samples_cnt;
    vec1i predictions;
    float accuracy = 0;

    bool selftest = false;
    bool print_info = true;

private:
    float sph_area = -1;

public:
    /*
     * Connectivity graph block
     */
    bool using_connectivity_feature = false;
    std::vector<std::map<int, float>> edges; // edges[from][to] = weight

    void initialize_connectivity_feature();
};
