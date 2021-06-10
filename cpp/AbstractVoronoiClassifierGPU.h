#pragma once

#include <string>
#include <random>
#include <chrono>
#include <functional>
#include <memory>
#include <fstream>

#include <libgpu/device.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>

#include "geometry.h"
#include "weights.h"
#include "mat2f.h"

/**
 * An object of this class is responsible for:
 *  - Train/test data reading
 *  - Precalculation of dxdx matrix
 *  - Generation of rays, precalculation of ur matrices
 *  - Partial handling of hitlog (except for actual writing)
 *
 * It does not include:
 *  - What exactly happens when the ray is casted
 */
class AbstractVoronoiClassifierGPU {
public:
    explicit AbstractVoronoiClassifierGPU(int seed = 239);

    virtual void raycast(int ch_id, int it, int test_cur_chunk_size, int test_index_offset) = 0;

    void load_train_data(const std::string &filename, int max_n = -1, bool nolabel = false);
    void load_test_data(const std::string &filename);

    /* Implementation specific method, doesn't include setting of a weight */
    void prepare(int argc, char **argv);

    virtual void init_kernels();

    /* Important to run one of the following two functions */
    void precalculate_dxdx();
    void load_dxdx(const std::string &filename);

    void save_dxdx(const std::string &filename);

    /* Use train data as test data as well (each test point tested against all other points) */
    void init_selftest();

    void perform_iterations_and_update(int niter, bool only_graph = false);
    void update_statistics();
    void reset_summaries();

    virtual void load_classification_data(const std::string &directory);
    virtual void save_classification_data(const std::string &directory);

protected:
    virtual void before_iterations();
    virtual void after_iterations();

protected:
    std::mt19937 random_engine;
    std::normal_distribution<float> normal_distribution;
    std::uniform_real_distribution<float> uniform_real_distribution;

private:
    bool has_next_rand_normal = false;
    float next_rand_normal;
    unsigned long long rand_state;

protected:
    float rand_normal();
    float rand_float();
    vec1f rand_vec_on_sphere(int ndim);

private:
    float sphere_area(int ndim);

    void precalculate_rays(int count);

    void load_test_chunk_gpu(int chunk_id);
    void load_train_chunk_gpu(int chunk_id);
    void load_dxdx_chunk_gpu(int chunk_id);

public:
    int train_n, test_n, d, k;
    mat2f train_data, test_data;
    vec1i train_labels, test_labels;

public:
    mat2f series;
    vec1i samples_cnt;
    vec1i predictions;
    float accuracy = 0;

    bool selftest = false;
    bool print_info = true;

protected:
    float sph_area = -1;

    mat2f dxdx_precalc;
    int test_max_chunk_size;
    int train_max_chunk_size;

    vec2f train_ur_precalc;
    vec2f test_ur_precalc;

protected:
    gpu::Device device;
    gpu::Context context;
    gpu::gpu_mem_32f train_data_gpu, test_data_gpu;
    gpu::gpu_mem_32i train_labels_gpu;
    gpu::gpu_mem_32f dxdx_precalc_chunk_gpu;
    gpu::gpu_mem_32f u_gpu;
    gpu::gpu_mem_32f train_ur_precalc_gpu;
    gpu::gpu_mem_32f test_ur_precalc_gpu;

private:
    int loaded_test_chunk_gpu = -1;
    int loaded_train_chunk_gpu = -1;
    int loaded_dxdx_chunk_gpu = -1;

    ocl::Kernel precalculate_dxdx_k;
    std::vector<size_t> precalculate_dxdx_group_size;
    ocl::Kernel precalculate_ur_k;
    std::vector<size_t> precalculate_ur_group_size;
    ocl::Kernel transpose_k;

private:
    ocl::Kernel init_kernel(const std::string &name, const std::string &defines = "");

protected:
    gpu::WorkSize work_size(unsigned int gsX, unsigned int wsX);
    gpu::WorkSize work_size(unsigned int gsX, unsigned int gsY, unsigned int wsX, unsigned int wsY);
    gpu::WorkSize work_size(unsigned int gsX, unsigned int gsY, unsigned int gsZ,
                            unsigned int wsX, unsigned int wsY, unsigned int wsZ);

public:
    bool log_all_hits = false;
    std::ofstream hit_log_os;
    void initialize_hits_logging(const std::string &log_filename);
};
