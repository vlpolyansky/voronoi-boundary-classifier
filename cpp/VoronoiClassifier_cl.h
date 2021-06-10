#pragma once

#include <string>
#include <random>
#include <chrono>
#include <functional>
#include <memory>

#include <libgpu/device.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include "AbstractVoronoiClassifierGPU.h"
#include "geometry.h"
#include "weights.h"
#include "mat2f.h"

class VoronoiClassifier : public AbstractVoronoiClassifierGPU {
public:
    explicit VoronoiClassifier(int seed = 239);

    void raycast(int ch_id, int it, int test_cur_chunk_size, int test_index_offset) override;

    void init_kernels() override;

    void set_weight(const std::shared_ptr<Weight> &weight);

    void load_classification_data(const std::string &directory) override;
    void save_classification_data(const std::string &directory) override;

    float series_to_integral_value(int i, int k_);

    void save_graph(const std::string &npy_filename);

public:
    std::shared_ptr<Weight> weight;

private:
    gpu::gpu_mem_32i best_i_gpu;
    gpu::gpu_mem_32f best_t_gpu;

    ocl::Kernel shoot_vector_k;

    ocl::Kernel init_kernel(const std::string &name, const std::string &defines = "");

public:
    /*
     * Connectivity graph block
     */
    bool using_connectivity_feature = false;
    std::vector<std::map<int, float>> edges; // edges[from][to] = weight

    void initialize_connectivity_feature();

public:
    /*
     * Piercing block
     */
    bool use_piercing = false;
    float piercing_prob = 0.0f;
    const int MAX_PIERCE_N = 5;
};
