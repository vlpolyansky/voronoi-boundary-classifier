#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>
#include <tuple>

#include <cnpy.h>
#include <libutils/timer.h>

#include "utils.h"
#include "geometry.h"

#include "AbstractVoronoiClassifierGPU.h"
#include "cl/avc_cl.h"

AbstractVoronoiClassifierGPU::AbstractVoronoiClassifierGPU(int seed) : rand_state(seed), random_engine(seed) {
    std::vector<gpu::Device> devices = gpu::enumDevices();
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found!");
    }
    std::cout << "Available devices:" << std::endl;
    for (auto &d : devices) {
        std::cout << "    " << d.name << std::endl;
    }
    device = devices[0];  // todo: make selection
    std::cout << "Using the device " << device.name << std::endl;

    context.init(device.device_id_opencl);
    context.activate();

    mat2f::MAX_CHUNK_SIZE = device.getFreeMemory() / sizeof(float) / 4; // todo: change
//    mat2f::MAX_CHUNK_SIZE = 318678630;
//    std::cout << mat2f::MAX_CHUNK_SIZE << std::endl;
}

void AbstractVoronoiClassifierGPU::load_train_data(const std::string &filename, int max_n, bool nolabels) {
    cnpy::NpyArray data_npy;
    if (!nolabels) {
        data_npy = cnpy::npz_load(filename, "data");
    } else {
        data_npy = cnpy::npy_load(filename);
    }
    std::cout << filename << std::endl;

    ensure(data_npy.shape.size() >= 2, "Data number of dimensions should be at least 2.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32-bit.");

    train_n = data_npy.shape[0];
    if (max_n >= 0 && train_n > max_n) {
        std::cout << "Data truncated from " << train_n << " to " << max_n << std::endl;
        train_n = max_n;
    }
    if (data_npy.shape.size() > 2) {
        std::cout << "Warning: data shape size is " + std::to_string(data_npy.shape.size()) + ", flattening..." << std::endl;
    }
    d = 1;
    for (int i = 1; i < data_npy.shape.size(); i++) {
        d *= data_npy.shape[i];
    }

    train_data = mat2f(train_n, d, false);
    train_max_chunk_size = train_data.get_chunk_rows();
    if (print_info)
        std::cout << "Train data split into " << train_data.get_chunks_num() << " chunks" << std::endl;
    train_data.fill_from({data_npy.data<float>(), data_npy.data<float>() + train_n * d});

    if (!nolabels) {
        cnpy::NpyArray labels_npy = cnpy::npz_load(filename, "labels");

        ensure(labels_npy.shape.size() == 1 ||
               (labels_npy.shape.size() == 2 && labels_npy.shape[1] == 1), "Labels number of dimensions should be 1.");
        ensure(labels_npy.word_size == sizeof(int), "Label word size should be 32-bit.");

        train_labels = labels_npy.as_vec<int>();
        train_labels.resize(train_n);
    } else {
        train_labels = vec<int>(train_n, 0);
    }

    k = *std::max_element(train_labels.begin(), train_labels.end()) + 1;

    train_data_gpu.resizeN(train_max_chunk_size * d);

    train_labels_gpu.resizeN(train_n);
    train_labels_gpu.writeN(train_labels.data(), train_n);

    sph_area = sphere_area(d - 1);
}

void AbstractVoronoiClassifierGPU::load_test_data(const std::string &filename) {
    cnpy::npz_t test_npz = cnpy::npz_load(filename);
    cnpy::NpyArray &data_npy = test_npz["data"];
    cnpy::NpyArray &labels_npy = test_npz["labels"];

    ensure(data_npy.shape.size() == 2, "Data number of dimensions be 2.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32-bit.");
    ensure(labels_npy.shape.size() == 1, "Labels number of dimensions should be 1.");
    ensure(labels_npy.word_size == sizeof(int), "Label word size should be 32-bit.");

    test_n = data_npy.shape[0];
    ensure(d == data_npy.shape[1], "Dimensionality of the data should be the same.");

    test_max_chunk_size = mat2f::predict_chunk_rows(std::max(d, train_n));   // one for test_data, other for dxdx
    test_data = mat2f(test_n, d, test_max_chunk_size);
    if (print_info)
        std::cout << "Test data split into " << test_data.get_chunks_num() << " chunks" << std::endl;
    test_data.fill_from({data_npy.data<float>(), data_npy.data<float>() + data_npy.num_vals});

    test_labels = labels_npy.as_vec<int>();

    test_data_gpu.resizeN(test_max_chunk_size * d);

    series = mat2f(test_n, k, .0f);
    samples_cnt = vec1i(test_n, 0);
    predictions = vec1i(test_n, 0);
    accuracy = 0;
}

void AbstractVoronoiClassifierGPU::precalculate_dxdx() {
    dxdx_precalc = mat2f(test_n, train_n, test_max_chunk_size);
    dxdx_precalc_chunk_gpu.resizeN(
            test_max_chunk_size * train_max_chunk_size);
    vec1f buffer(test_max_chunk_size * train_max_chunk_size);
    for (int test_chunk = 0; test_chunk < test_data.get_chunks_num(); test_chunk++) {
        load_test_chunk_gpu(test_chunk);
        for (int train_chunk = 0; train_chunk < train_data.get_chunks_num(); train_chunk++) {
            load_train_chunk_gpu(train_chunk);
            timer precalculate_dxdx_timer;
            if (print_info)
                std::cout << "Chunk (" << test_chunk + 1 << ", " << train_chunk + 1 << ") out of ("
                          << test_data.get_chunks_num() << ", " << train_data.get_chunks_num() << ")... ";
            precalculate_dxdx_k.exec(
                    work_size(precalculate_dxdx_group_size[0], precalculate_dxdx_group_size[1],
                              precalculate_dxdx_group_size[2],
                              precalculate_dxdx_group_size[0], test_max_chunk_size, train_max_chunk_size),
                    test_data_gpu, train_data_gpu, dxdx_precalc_chunk_gpu,
                    test_data.get_chunk_rows(test_chunk), train_data.get_chunk_rows(train_chunk), d
            );
            if (print_info)
                std::cout << "(time: " << precalculate_dxdx_timer.elapsed() << ")" << std::endl;
            dxdx_precalc_chunk_gpu.readN(buffer.data(),
                                         test_data.get_chunk_rows(test_chunk) * train_data.get_chunk_rows(train_chunk));
            for (int i = 0; i < test_data.get_chunk_rows(test_chunk); i++) {
                for (int j = 0; j < train_data.get_chunk_rows(train_chunk); j++) {
                    dxdx_precalc(test_max_chunk_size * test_chunk + i, train_max_chunk_size * train_chunk + j) =
                            buffer[i * train_data.get_chunk_rows(train_chunk) + j];
                }
            }
        }
    }
    if (print_info)
        std::cout << "dxdx consists of " << dxdx_precalc.get_chunks_num() << " chunks, "
                  << dxdx_precalc.get_chunk_rows(0) << " rows each" << std::endl;

    dxdx_precalc_chunk_gpu.resizeN(test_max_chunk_size * train_n);
}

void AbstractVoronoiClassifierGPU::load_dxdx(const std::string &filename) {
    cnpy::npz_t dxdx_npz = cnpy::npz_load(filename);
    dxdx_precalc = mat2f(test_n, train_n, test_max_chunk_size);
    int offset = 0;
    for (int i = 0;; i++) {
        if (dxdx_npz.find(std::to_string(i)) == dxdx_npz.end()) {
            ensure(offset == test_n, "Bad dxdx file.");
            break;
        }
        cnpy::NpyArray &npy = dxdx_npz[std::to_string(i)];
        ensure(offset + npy.shape[0] <= test_n, "Bad dxdx file.");
        ensure(npy.shape[1] == train_n, "Bad dxdx file.");
        ensure(npy.word_size == sizeof(float), "dxdx data word size should be 32-bit.");
        offset = dxdx_precalc.fill_submat_from({npy.data<float>(), npy.data<float>() + npy.num_vals}, offset);
    }
    std::cout << "dxdx consists of " << dxdx_precalc.get_chunks_num() << " blocks, "
              << dxdx_precalc.get_chunk_rows(0) << " rows each" << std::endl;

    dxdx_precalc_chunk_gpu.resizeN(test_max_chunk_size * train_n);
}

void AbstractVoronoiClassifierGPU::save_dxdx(const std::string &filename) {
    std::cout << "Saving dxdx to a file" << std::endl;
    for (int i = 0; i < dxdx_precalc.get_chunks_num(); i++) {
        cnpy::npz_save(filename, std::to_string(i), dxdx_precalc.get_data(i).data(),
                       {unsigned(dxdx_precalc.get_chunk_rows(i)), unsigned(dxdx_precalc.cols)}, i == 0 ? "w" : "a");
    }
}

void AbstractVoronoiClassifierGPU::init_selftest() {
    selftest = true;

    test_n = train_n;

    test_max_chunk_size = mat2f::predict_chunk_rows(std::max(d, train_n));   // one for test_data, other for dxdx
    test_data = mat2f(test_n, d, test_max_chunk_size);
    if (print_info)
        std::cout << "Test data split into " << test_data.get_chunks_num() << " chunks" << std::endl;
    for (int i = 0; i < train_data.get_chunks_num(); i++) {
        test_data.fill_submat_from(train_data.get_data(i), i * train_max_chunk_size);
    }

    test_labels = train_labels;

    test_data_gpu.resizeN(test_max_chunk_size * d);

    series = mat2f(test_n, k, .0f);
    samples_cnt = vec1i(test_n, 0);
    predictions = vec1i(test_n, 0);
    accuracy = 0;
}


void AbstractVoronoiClassifierGPU::perform_iterations_and_update(int niter, bool only_graph) {

    precalculate_rays(niter);

    before_iterations();

    test_ur_precalc_gpu.resizeN(test_max_chunk_size);
    train_ur_precalc_gpu.resizeN(train_n);

    gpu::gpu_mem_32f tmp;
    for (int ch_id = 0; ch_id < dxdx_precalc.get_chunks_num(); ch_id++) {
        vec1f &dxdx_chunk = dxdx_precalc.get_data(ch_id);
        load_dxdx_chunk_gpu(ch_id);
        int test_index_offset = ch_id * test_max_chunk_size;
        int test_cur_chunk_size = test_data.get_chunk_rows(ch_id);

        tmp.resize(dxdx_precalc_chunk_gpu.size());
        if (dxdx_chunk.size() != dxdx_precalc.get_chunk_rows(ch_id) * dxdx_precalc.cols)
            std::cerr << "something went wrong..." << std::endl;
        transpose_k.exec(work_size(16, 16, dxdx_precalc.get_chunk_rows(ch_id), dxdx_precalc.cols),
                         dxdx_precalc_chunk_gpu, tmp, dxdx_precalc.get_chunk_rows(ch_id), dxdx_precalc.cols);
        std::swap(tmp, dxdx_precalc_chunk_gpu);

        timer total_t;
        for (int it = 0; it < niter; it++) {
            test_ur_precalc_gpu.writeN(test_ur_precalc[it].data() + test_index_offset,
                                       test_cur_chunk_size);
            train_ur_precalc_gpu.writeN(train_ur_precalc[it].data(), train_n);

            raycast(ch_id, it, test_cur_chunk_size, test_index_offset);
        }

        std::swap(tmp, dxdx_precalc_chunk_gpu);

        if (ch_id == 0 && print_info) {
            std::cout << "  Time on the first chunk: " << total_t.elapsed() << std::endl;
        }

    }

    after_iterations();

    update_statistics();
    if (log_all_hits) {
        hit_log_os.flush();
    }
}

void AbstractVoronoiClassifierGPU::before_iterations() {
}

void AbstractVoronoiClassifierGPU::after_iterations() {
}

void AbstractVoronoiClassifierGPU::update_statistics() {
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

void AbstractVoronoiClassifierGPU::reset_summaries() {
    series.reset();
    std::fill(samples_cnt.begin(), samples_cnt.end(), 0);
    std::fill(predictions.begin(), predictions.end(), 0);
    accuracy = 0;
}

void AbstractVoronoiClassifierGPU::load_classification_data(const std::string &directory) {
    std::cout << "Loading data from " << directory << std::endl;
    series = load_mat2d(directory + "/series.txt", test_n, k);
    samples_cnt = load_vec1i(directory + "/samples_cnt.txt", test_n);
    update_statistics();
}

void AbstractVoronoiClassifierGPU::save_classification_data(const std::string &directory) {
    std::cout << "Saving results into " << directory << std::endl;
    save_mat2d(series, directory + "/series.txt", false);
    save_vector(samples_cnt, directory + "/samples_cnt.txt", false);
    save_vector(predictions, directory + "/predictions.txt", false);

    if (log_all_hits) {
        hit_log_os.close();
    }
}

float AbstractVoronoiClassifierGPU::rand_normal() {
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

float AbstractVoronoiClassifierGPU::rand_float() {
//    return uniform_real_distribution(random_engine);
    return (1.0f * random_engine() - random_engine.min()) / (random_engine.max() - random_engine.min());
//    rand_state = (rand_state * 16807) & 2147483647;
//    return 1.0f * rand_state / 2147483647;
}

vec1f AbstractVoronoiClassifierGPU::rand_vec_on_sphere(int ndim) {
    vec1f a(ndim);
    for (int i = 0; i < ndim; i++) {
        a[i] = rand_normal();
    }
    a = a / length(a);
    return a;
}

// ndim is the dimensionality of a sphere, not of space!
float AbstractVoronoiClassifierGPU::sphere_area(int ndim) {
    float v = 1;
    float s = 2;
    for (int i = 1; i <= ndim; i++) {
        float new_v = s / i;
        s = 2 * PI * v;
        v = new_v;
    }
    return s;
}

void AbstractVoronoiClassifierGPU::precalculate_rays(int count) {
    timer total_t;
    timer rays_t;
    vec2f rays;
    for (int i = 0; i < count; i++) {
        rays.push_back(rand_vec_on_sphere(d));
    }
    u_gpu.resizeN(d);
    if (print_info)
        std::cout << "  Rays precalculation: " << "rays gen: " << rays_t.elapsed();

    timer precalc_t;
    test_ur_precalc = make_vec2d(count, test_n);
    test_ur_precalc_gpu.resizeN(test_max_chunk_size);
    gpu::gpu_mem_32f tmp;
    for (int i = 0; i < test_data.get_chunks_num(); i++) {
        load_test_chunk_gpu(i);
        tmp.resize(test_data_gpu.size());
        transpose_k.exec(work_size(16, 16, test_max_chunk_size, test_data.cols),
                         test_data_gpu, tmp, test_data.get_chunk_rows(i), test_data.cols);
        std::swap(test_data_gpu, tmp);

        for (int j = 0; j < count; j++) {
            u_gpu.writeN(rays[j].data(), d);
            precalculate_ur_k.exec(
                    work_size(precalculate_ur_group_size[0], precalculate_ur_group_size[1],
                              test_max_chunk_size, precalculate_ur_group_size[1]),
                    test_data_gpu, u_gpu, test_ur_precalc_gpu,
                    test_data.get_chunk_rows(i), d);
            test_ur_precalc_gpu.readN(
                    test_ur_precalc[j].data() + i * test_max_chunk_size, test_data.get_chunk_rows(i));
        }

        std::swap(test_data_gpu, tmp);
    }

    train_ur_precalc = make_vec2d(count, train_n);
    train_ur_precalc_gpu.resizeN(train_max_chunk_size);
    for (int i = 0; i < train_data.get_chunks_num(); i++) {
        load_train_chunk_gpu(i);
        tmp.resize(train_data_gpu.size());
        transpose_k.exec(work_size(16, 16, train_max_chunk_size, train_data.cols),
                         train_data_gpu, tmp, train_data.get_chunk_rows(i), train_data.cols);
        std::swap(train_data_gpu, tmp);

        for (int j = 0; j < count; j++) {
            u_gpu.writeN(rays[j].data(), d);
            precalculate_ur_k.exec(
                    work_size(precalculate_ur_group_size[0], precalculate_ur_group_size[1],
                              train_max_chunk_size, precalculate_ur_group_size[1]),
                    train_data_gpu, u_gpu, train_ur_precalc_gpu,
                    train_data.get_chunk_rows(i), d);
            train_ur_precalc_gpu.readN(
                    train_ur_precalc[j].data() + i * train_max_chunk_size, train_data.get_chunk_rows(i));
        }

        std::swap(train_data_gpu, tmp);
    }

    test_ur_precalc_gpu.resizeN(test_max_chunk_size);
    train_ur_precalc_gpu.resizeN(train_n);

    if (print_info)
        std::cout << ", precalc time: " << precalc_t.elapsed() << ", total: " << total_t.elapsed() << std::endl;
}

void AbstractVoronoiClassifierGPU::load_test_chunk_gpu(int chunk_id) {
    if (loaded_test_chunk_gpu != chunk_id) {
        loaded_test_chunk_gpu = chunk_id;
        vec1f &data = test_data.get_data(chunk_id);
        test_data_gpu.writeN(data.data(), data.size());
    }
}

void AbstractVoronoiClassifierGPU::load_train_chunk_gpu(int chunk_id) {
    if (loaded_train_chunk_gpu != chunk_id) {
        loaded_train_chunk_gpu = chunk_id;
        vec1f &data = train_data.get_data(chunk_id);
        train_data_gpu.writeN(data.data(), data.size());
    }
}

void AbstractVoronoiClassifierGPU::load_dxdx_chunk_gpu(int chunk_id) {
    if (loaded_dxdx_chunk_gpu != chunk_id) {
        loaded_dxdx_chunk_gpu = chunk_id;
        vec1f &data = dxdx_precalc.get_data(chunk_id);
        dxdx_precalc_chunk_gpu.writeN(data.data(), data.size());
    }
}

ocl::Kernel AbstractVoronoiClassifierGPU::init_kernel(const std::string &name, const std::string &defines) {
    return ocl::Kernel(avc_kernel_sources, avc_kernel_sources_length, name, defines);
}

gpu::WorkSize AbstractVoronoiClassifierGPU::work_size(unsigned int gsX, unsigned int wsX) {
    return {gsX, (wsX + gsX - 1) / gsX * gsX};
}

gpu::WorkSize AbstractVoronoiClassifierGPU::work_size(unsigned int gsX, unsigned int gsY, unsigned int wsX, unsigned int wsY) {
    return {gsX, gsY, (wsX + gsX - 1) / gsX * gsX, (wsY + gsY - 1) / gsY * gsY};
}

gpu::WorkSize
AbstractVoronoiClassifierGPU::work_size(unsigned int gsX, unsigned int gsY, unsigned int gsZ, unsigned int wsX, unsigned int wsY,
                             unsigned int wsZ) {
    return {gsX, gsY, gsZ,
            (wsX + gsX - 1) / gsX * gsX, (wsY + gsY - 1) / gsY * gsY, (wsZ + gsZ - 1) / gsZ * gsZ};
}

void AbstractVoronoiClassifierGPU::initialize_hits_logging(const std::string &log_filename) {
    log_all_hits = true;
    hit_log_os = std::ofstream(log_filename);
    std::cout << "Hits logging: enabled, file: " << log_filename << std::endl;

}

std::vector<size_t> optimal_precalculate_dxdx_item_sizes(size_t max_work_group_size,
                                                         const std::vector<size_t> &max_work_item_sizes,
                                                         int train_n, int test_n, int d) {
    std::vector<size_t> sizes(3, 1);
    size_t cur_total = 1;
    while (sizes[0] < max_work_item_sizes[0] && cur_total < max_work_group_size && sizes[0] * sizes[0] * 4 <= d) {
        sizes[0] <<= 1;
        cur_total <<= 1;
    }
    while (sizes[1] < max_work_item_sizes[1] && sizes[2] < max_work_item_sizes[2] &&
           cur_total * 2 < max_work_group_size) {
        sizes[1] <<= 1;
        sizes[2] <<= 1;
        cur_total <<= 2;
    }

    return sizes;
}

std::vector<size_t> optimal_precalculate_ur_item_sizes(size_t max_work_group_size,
                                                       const std::vector<size_t> &max_work_item_sizes,
                                                       int train_n, int test_n, int d) {
    std::vector<size_t> sizes(2, 1);
    size_t cur_total = 1;
    while (sizes[1] < max_work_item_sizes[1] && cur_total < max_work_group_size && sizes[1] * sizes[1] * 4 <= d) {
        sizes[1] <<= 1;
        cur_total <<= 1;
    }
    while (sizes[0] < max_work_item_sizes[0] && cur_total < max_work_group_size) {
        sizes[0] <<= 1;
        cur_total <<= 1;
    }

    return sizes;
}

void AbstractVoronoiClassifierGPU::init_kernels() {
    size_t max_work_group_size = context.getMaxWorkgroupSize();
    max_work_group_size = 256; // todo fix for cpu
    std::vector<size_t> max_work_item_sizes = context.getMaxWorkItemSizes();

    precalculate_dxdx_group_size = optimal_precalculate_dxdx_item_sizes(max_work_group_size, max_work_item_sizes,
                                                                        train_max_chunk_size, test_max_chunk_size, d);
    if (print_info)
        std::cout << "precalculate_dxdx_group_size: ["
                  << precalculate_dxdx_group_size[0] << ", "
                  << precalculate_dxdx_group_size[1] << ", "
                  << precalculate_dxdx_group_size[2] << "]" << std::endl;
    precalculate_dxdx_k = init_kernel(
            "precalculate_dxdx",
            "-D PRECALCULATE_DXDX_WORKGROUP_SIZE=" + std::to_string(precalculate_dxdx_group_size[0])
            + " -D PRECALCULATE_DXDX_WORKGROUP_SIZE_GRID=" + std::to_string(precalculate_dxdx_group_size[1]));

    precalculate_ur_group_size = optimal_precalculate_ur_item_sizes(max_work_group_size, max_work_item_sizes,
                                                                    train_max_chunk_size, test_max_chunk_size, d);
    if (print_info)
        std::cout << "precalculate_ur_group_size: ["
                  << precalculate_ur_group_size[0] << ", "
                  << precalculate_ur_group_size[1] << "]" << std::endl;
    precalculate_ur_k = init_kernel(
            "precalculate_ur",
            "-D PRECALCULATE_UR_WORKGROUP_SIZE_GRID=" + std::to_string(precalculate_ur_group_size[0])
            + " -D PRECALCULATE_UR_WORKGROUP_SIZE=" + std::to_string(precalculate_ur_group_size[1]));

    transpose_k = init_kernel("transpose");
}

void AbstractVoronoiClassifierGPU::prepare(int argc, char **argv) {
    if (print_info)
        std::cout << "Initializing kernels" << std::endl;
    init_kernels();
    char *dxdx_filename = get_cmd_option(argv, argv + argc, "--dxdx");
    if (dxdx_filename) {
        if (print_info)
            std::cout << "Loading dxdx matrix" << std::endl;
        load_dxdx(dxdx_filename);
    } else {
        if (print_info)
            std::cout << "Calculating dxdx matrix" << std::endl;
        timer t;
        precalculate_dxdx();
        if (print_info)
            std::cout << "Precalculation took " << t.elapsed() << " seconds" << std::endl;
    }
}