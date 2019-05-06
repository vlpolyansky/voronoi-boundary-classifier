#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>
#include <tuple>

#include <cnpy.h>
#include <libutils/timer.h>

#include "utils.h"
#include "geometry.h"
#include "VoronoiClassifier_kd.h"


#include <omp.h>


VoronoiClassifier::VoronoiClassifier(int seed) : rand_state(seed), random_engine(seed) {
//    mat2f::MAX_CHUNK_SIZE = device.getFreeMemory() / sizeof(float) / 4;
    mat2f::MAX_CHUNK_SIZE = 1 << 24;
//    std::cout << mat2f::MAX_CHUNK_SIZE << std::endl;
}

void VoronoiClassifier::load_train_data(const std::string &filename, int max_n) {
    cnpy::npz_t train_npz = cnpy::npz_load(filename);
    cnpy::NpyArray &data_npy = train_npz["data"];
    cnpy::NpyArray &labels_npy = train_npz["labels"];

    ensure(data_npy.shape.size() == 2, "Data number of dimensions be 2.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32-bit.");
    ensure(labels_npy.shape.size() == 1, "Labels number of dimensions should be 1.");
    ensure(labels_npy.word_size == sizeof(int), "Label word size should be 32-bit.");

    train_n = data_npy.shape[0];
    if (max_n >= 0 && train_n > max_n) {
        std::cout << "Data truncated from " << train_n << " to " << max_n << std::endl;
        train_n = max_n;
    }
    d = data_npy.shape[1];

    train_data = mat2f(train_n, d, false);
    train_max_chunk_size = train_data.get_chunk_rows();
    if (print_info)
        std::cout << "Train data split into " << train_data.get_chunks_num() << " chunks" << std::endl;
    train_data.fill_from({data_npy.data<float>(), data_npy.data<float>() + train_n * d});

    train_labels = labels_npy.as_vec<int>();
    train_labels.resize(train_n);
    k = *std::max_element(train_labels.begin(), train_labels.end()) + 1;

    sph_area = sphere_area(d - 1);
}

void VoronoiClassifier::load_test_data(const std::string &filename) {
    cnpy::npz_t test_npz = cnpy::npz_load(filename);
    cnpy::NpyArray &data_npy = test_npz["data"];
    cnpy::NpyArray &labels_npy = test_npz["labels"];

    ensure(data_npy.shape.size() == 2, "Data number of dimensions be 2.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32-bit.");
    ensure(labels_npy.shape.size() == 1, "Labels number of dimensions should be 1.");
    ensure(labels_npy.word_size == sizeof(int), "Label word size should be 32-bit.");

    test_n = data_npy.shape[0];
    ensure(d == data_npy.shape[1], "Dimensionality of the data should be the same.");

    test_data = mat2f(test_n, d, false);
    test_max_chunk_size = test_data.get_chunk_rows();
    if (print_info)
        std::cout << "Test data split into " << test_data.get_chunks_num() << " chunks" << std::endl;
    test_data.fill_from({data_npy.data<float>(), data_npy.data<float>() + data_npy.num_vals});

    test_labels = labels_npy.as_vec<int>();

    series = mat2f(test_n, k, .0f);
    samples_cnt = vec1i(test_n, 0);
    predictions = vec1i(test_n, 0);
    accuracy = 0;

}

void VoronoiClassifier::init_selftest() {
    selftest = true;

    test_n = train_n;

    test_data = mat2f(test_n, d, false);
    test_max_chunk_size = test_data.get_chunk_rows();
    if (print_info)
        std::cout << "Test data split into " << test_data.get_chunks_num() << " chunks" << std::endl;
    for (int i = 0; i < train_data.get_chunks_num(); i++) {
        test_data.fill_submat_from(train_data.get_data(i), i * train_max_chunk_size);
    }

    test_labels = train_labels;

    series = mat2f(test_n, k, .0f);
    samples_cnt = vec1i(test_n, 0);
    predictions = vec1i(test_n, 0);
    accuracy = 0;
}

void VoronoiClassifier::perform_iterations_and_update(int niter) {

    timer t;
    for (int it = 0; it < niter; it++) {
        vec1f u = rand_vec_on_sphere(d);
#pragma omp parallel for
        for (int i = 0; i < test_n; i++) {
            cast_ray(i, u);
        }
    }
    std::cout << "Iteration took: " << t.elapsed() << " sec" << std::endl;

    update_statistics();
}

void VoronoiClassifier::cast_ray(int test_idx, const range<float> &u) {
    const range<float> &test_p = test_data[test_idx];
    float l = 0;
    float r = 100;      // todo: replace with max distance
    float niter = 10;   // todo: replace with something better
    float eps = 1e-5;
    bool hit = false;
    for (int it = 0; it < niter && r - l > eps; it++) {
        float m = (l + r) / 2;
        std::vector<float> p = test_p + m * u;
        float dist;
        int train_idx = nn(p, &dist, m * m);
        if (m < dist || (selftest && test_idx == train_idx)) {
            l = m;
        } else {
            hit = true;
            const range<float> &train_p = train_data[train_idx];
            vec1f ab = train_p - test_p;
            float t = 0.5f * (ab * ab) / (u * ab);
            r = std::min(t, r);  // due to approximate nn search, m can be larger than r
        }
    }

    if (hit) {
        std::vector<float> p = test_p + r * u;
        float dist;
        int train_idx = nn(p, &dist);
        const range<float> &train_p = train_data[train_idx];
        vec1f ab = train_p - test_p;
        float sin_beta = std::abs(u * ab) / std::sqrt(ab * ab);
        series(test_idx, train_labels[train_idx]) +=
                1.0f / (sin_beta * r) * weight->estimate(r);
        samples_cnt[test_idx]++;

        if (using_connectivity_feature) {
            // Calculate number of hits
            std::map<int, float> &edge_set = edges[test_idx];
            auto e = edge_set.find(train_idx);
            if (e != edge_set.end()) {
                e->second += 1.0f;
            } else {
                edge_set[train_idx] = 1.0f;
            }
        }
    }
}

int VoronoiClassifier::nn(const range<float> &x, float *distance, float margin_sqr) {
//    int best_i = 0;
//    float best_d = length_sqr(x - train_data[0]);
//
//    for (int i = 1; i < train_n; i += 1) {
//        float cur_d = length_sqr(x - train_data[i]);
//        if (cur_d < best_d) {
//            best_d = cur_d;
//            best_i = i;
//        }
//    }
//
//    *distance = std::sqrt(best_d);
//    return best_i;
    float tmp;
    if (kd_tree->find_nn(x, &tmp, margin_sqr) == -1) {
        std::cout << "HERE" << std::endl;
    }
    int res = kd_tree->find_nn(x, distance, margin_sqr);
    *distance = std::sqrt(*distance);
    return res;
}

void VoronoiClassifier::update_statistics() {
    int sum = 0;
    for (int i = 0; i < test_n; i++) {
        auto range = series[i];
        predictions[i] = static_cast<int>(
                std::max_element(range.begin, range.end) - range.begin);
        if (predictions[i] == test_labels[i]) {
            sum++;
        }
    }
    accuracy = 1.f * sum / test_n;
}

void VoronoiClassifier::set_weight(const std::shared_ptr<Weight> &weight) {
    VoronoiClassifier::weight = weight;
}

float VoronoiClassifier::rand_normal() {
//    return normal_distribution(random_engine);
    // Marsaglia polar method
    if (has_next_rand_normal) {
        has_next_rand_normal = false;
        return next_rand_normal;
    }
    float u, v, s, t;
    do {
        u = rand_float() * 2.0f - 1.0f;
        v = rand_float() * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s == 0 || s >= 1);
    t = std::sqrt(-2.0f * std::log(s) / s);
    has_next_rand_normal = true;
    next_rand_normal = v * t;
    return u * t;
}

float VoronoiClassifier::rand_float() {
//    return uniform_real_distribution(random_engine);
    return (1.0f * random_engine() - random_engine.min()) / (random_engine.max() - random_engine.min());
//    rand_state = (rand_state * 16807) & 2147483647;
//    return 1.0f * rand_state / 2147483647;
}

vec1f VoronoiClassifier::rand_vec_on_sphere(int ndim) {
    vec1f a(ndim);
    for (int i = 0; i < ndim; i++) {
        a[i] = rand_normal();
    }
    a = a / length(a);
    return a;
}

// ndim is the dimensionality of a sphere, not of space!
float VoronoiClassifier::sphere_area(int ndim) {
    float v = 1;
    float s = 2;
    for (int i = 1; i <= ndim; i++) {
        float new_v = s / i;
        s = 2 * PI * v;
        v = new_v;
    }
    return s;
}

float VoronoiClassifier::series_to_integral_value(int test_idx, int label_idx) {
    return sph_area * series[test_idx][label_idx] / samples_cnt[test_idx];
}

void VoronoiClassifier::initialize_connectivity_feature() {
    edges = std::vector<std::map<int, float>>(test_n);
    using_connectivity_feature = true;
    std::cout << "Gathering connectivity information: enabled" << std::endl;
}

void VoronoiClassifier::load_classification_data(const std::string &directory) {
    std::cout << "Loading data from " << directory << std::endl;
    series = load_mat2d(directory + "/series.txt", test_n, k);
    samples_cnt = load_vec1i(directory + "/samples_cnt.txt", test_n);
    update_statistics();

    if (using_connectivity_feature) {
        std::ifstream in(directory + "/edges.txt");
        for (int from = 0; from < test_n; from++) {
            int c;
            in >> c;
            for (int j = 0; j < c; j++) {
                int to;
                float weight;
                in >> to >> weight;
                edges[from][to] = weight;
            }
        }
        in.close();
    }
}

void VoronoiClassifier::save_classification_data(const std::string &directory) {
    std::cout << "Saving results into " << directory << std::endl;
    save_mat2d(series, directory + "/series.txt", false);
    save_vector(samples_cnt, directory + "/samples_cnt.txt", false);
    save_vector(predictions, directory + "/predictions.txt", false);

    if (using_connectivity_feature) {
        std::ofstream out(directory + "/edges.txt");
        for (int from = 0; from < test_n; from++) {
            out << edges[from].size();
            for (const auto &e: edges[from]) {
                out << " " << e.first << " " << e.second;
            }
            out << "\n";
        }
        out.close();
    }
}

void VoronoiClassifier::prepare(int argc, char **argv) {
    kd_tree = std::make_shared<KDTree>(train_data);
    if (print_info)
        std::cout << "Building the K-d tree" << std::endl;
    kd_tree->init();
}
