#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>
#include <tuple>

#include <cnpy.h>
#include <libutils/timer.h>

#include "utils.h"
#include "geometry.h"
#include "VoronoiClassifier_cl.h"

#include "cl/voronoi_cl.h"


VoronoiClassifier::VoronoiClassifier(int seed) : AbstractVoronoiClassifierGPU(seed) { }

void VoronoiClassifier::raycast(int ch_id, int it, int test_cur_chunk_size, int test_index_offset) {
    unsigned int workGroupSize = 128;

    best_i_gpu.resizeN(test_max_chunk_size);
    best_t_gpu.resizeN(test_max_chunk_size);

    vec1i best_i(test_cur_chunk_size, -1);
    vec1f best_t(test_cur_chunk_size, 0);
    best_i_gpu.writeN(best_i.data(), test_cur_chunk_size);

    int pierce_n = 0;
    if (use_piercing) {
        pierce_n = MAX_PIERCE_N;
        for (int i = 0; i < MAX_PIERCE_N; i++) {
            if (rand_float() <= 1.0f - piercing_prob) {
                pierce_n = i;
                break;
            }
        }
    }

    timer sh_t;
    if (selftest) {
        shoot_vector_k.exec(
                work_size(workGroupSize, test_max_chunk_size),
                dxdx_precalc_chunk_gpu, test_ur_precalc_gpu, train_ur_precalc_gpu,
                pierce_n,
                best_i_gpu, best_t_gpu,
                dxdx_precalc.get_chunk_rows(ch_id), dxdx_precalc.cols,
                test_index_offset
        );
    } else {
        shoot_vector_k.exec(
                work_size(workGroupSize, test_max_chunk_size),
                dxdx_precalc_chunk_gpu, test_ur_precalc_gpu, train_ur_precalc_gpu,
                pierce_n,
                best_i_gpu, best_t_gpu,
                dxdx_precalc.get_chunk_rows(ch_id), dxdx_precalc.cols
        );
    }

    best_i_gpu.readN(best_i.data(), test_cur_chunk_size);
    best_t_gpu.readN(best_t.data(), test_cur_chunk_size);

    for (int i = 0; i < test_cur_chunk_size; i++) {
        int global_test_idx = test_index_offset + i;
        int train_idx = best_i[i];
        if (train_idx >= 0) {
            float sin_beta =
                    std::abs(train_ur_precalc[it][train_idx] - test_ur_precalc[it][global_test_idx])
                    / std::sqrt(dxdx_precalc(global_test_idx, train_idx));
            // IMPORTANT! The formula should contain std::pow(best_t[i], d),
            // but as a part of optimization we assume that the weight function also
            // implicitly includes std::pow(best_t[i], -d).
            series(global_test_idx, train_labels[train_idx]) +=
                    1.0f / (best_t[i] * sin_beta) * weight->estimate(best_t[i]);
            samples_cnt[global_test_idx]++;

            if (using_connectivity_feature) {
                // Calculate number of hits
                std::map<int, float> &edge_set = edges[global_test_idx];
                auto e = edge_set.find(train_idx);
                if (e != edge_set.end()) {
                    e->second += 1.0f;
                } else {
                    edge_set[train_idx] = 1.0f;
                }
            }
            if (log_all_hits) {
                hit_log_os.write(reinterpret_cast<const char*>(&global_test_idx), sizeof(int));
                hit_log_os.write(reinterpret_cast<const char*>(&train_idx), sizeof(int));
                hit_log_os.write(reinterpret_cast<const char*>(&best_t[i]), sizeof(float));
                hit_log_os.write(reinterpret_cast<const char*>(&sin_beta), sizeof(float));
            }
        } else {
            if (log_all_hits) {
                float sin_beta = 1;
                hit_log_os.write(reinterpret_cast<const char*>(&global_test_idx), sizeof(int));
                hit_log_os.write(reinterpret_cast<const char*>(&train_idx), sizeof(int));
                hit_log_os.write(reinterpret_cast<const char*>(&best_t[i]), sizeof(float));
                hit_log_os.write(reinterpret_cast<const char*>(&sin_beta), sizeof(float));
            }
        }
    }
}

void VoronoiClassifier::set_weight(const std::shared_ptr<Weight> &weight) {
    VoronoiClassifier::weight = weight;
}

float VoronoiClassifier::series_to_integral_value(int test_idx, int label_idx) {
    return sph_area * series[test_idx][label_idx] / samples_cnt[test_idx];
}


ocl::Kernel VoronoiClassifier::init_kernel(const std::string &name, const std::string &defines) {
    return ocl::Kernel(kernel_sources, kernel_sources_length, name, defines);
}

void VoronoiClassifier::init_kernels() {
    AbstractVoronoiClassifierGPU::init_kernels();

    shoot_vector_k = init_kernel("shoot_piercing_vector", selftest ? "-D SELFTEST" : "");
}

void VoronoiClassifier::initialize_connectivity_feature() {
    edges = std::vector<std::map<int, float>>(test_n);
    using_connectivity_feature = true;
    std::cout << "Gathering connectivity information: enabled" << std::endl;
}

void VoronoiClassifier::load_classification_data(const std::string &directory) {
    AbstractVoronoiClassifierGPU::load_classification_data(directory);

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
    AbstractVoronoiClassifierGPU::save_classification_data(directory);

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

void VoronoiClassifier::save_graph(const std::string &npy_filename) {
    std::cout << "Saving to " << npy_filename << std::endl;
    vec<int> output;  // stores from, to, significance

    for (int from = 0; from < test_n; from++) {
        for (const auto &e: edges[from]) {
            int to = e.first;
            int significance = static_cast<int>(e.second); // careful, we expect weight as integer!
            if (from < to) {
                auto other = edges[to].find(from);
                if (other != edges[to].end()) {
                    significance += other->second;
                }
                output.push_back(from);
                output.push_back(to);
                output.push_back(significance);
            }
        }
    }

    cnpy::npy_save(npy_filename, &output[0], {output.size() / 3, 3}, "w");
}