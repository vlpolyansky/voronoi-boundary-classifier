#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>

#include <libutils/timer.h>

#include "utils.h"
#include "geometry.h"
#include "weights.h"

#define CL 1
#define KD 2

#ifndef VCTYPE
#define VCTYPE CL
#endif

#if VCTYPE == CL
#include "VoronoiClassifier_cl.h"
#elif VCTYPE == KD
#include "VoronoiClassifier_kd.h"
#else
#error Unknown implementation type
#endif

#if VCTYPE == CL or VCTYPE == KD
void set_weight(VoronoiClassifier& clf, int argc, char **argv) {
    std::string weight_type = get_cmd_option_string(argv, argv + argc, "--weight", "gpw");
    std::shared_ptr<Weight> weight;
    if (weight_type == "gpw") {
        float w_sigma = get_cmd_option_float(argv, argv + argc, "--wsigma", 1.0f);
        float w_p = get_cmd_option_float(argv, argv + argc, "--wp", 0.0f);
        float w_scale = get_cmd_option_float(argv, argv + argc, "--wscale", 1.0f);
        weight.reset(new GaussianPolynomialWeight(clf.d, w_p, w_sigma, w_scale));
    } else if (weight_type == "gcw") {
        float w_sigma = get_cmd_option_float(argv, argv + argc, "--wsigma", 1.0f);
        weight.reset(new GaussianConicalWeight(clf.d, w_sigma));
    } else if (weight_type == "thres") {
        float w_thres = get_cmd_option_float(argv, argv + argc, "--wthres", 1e9);
        weight.reset(new ThresholdWeight(clf.d, w_thres));
    } else {
        throw std::runtime_error("Unknown weight function: " + weight_type);
    }
    clf.set_weight(weight);
}
#endif

void run_classification(int argc, char **argv) {
    assert(argc >= 3);
    auto out_dir = init_out_dir(argc, argv);

    std::string train_filename = argv[1];

    int seed = get_cmd_option_int(argv, argv + argc, "--seed", 239);

    VoronoiClassifier clf(seed);
    bool silent = false;
    if (cmd_option_exists(argv, argv + argc, "--silent")) {
        silent = true;
        clf.print_info = false;
    }
    if (!silent)
        std::cout << "Reading train data" << std::endl;
    clf.load_train_data(train_filename);
    if (cmd_option_exists(argv, argv + argc, "--selftest")) {
        if (!silent)
            std::cout << "Initializing selftest" << std::endl;
        clf.init_selftest();
    } else {
        if (!silent)
            std::cout << "Reading test data" << std::endl;
        std::string test_filename = argv[2];
        clf.load_test_data(test_filename);
    }
#if VCTYPE == CL or VCTYPE == KD
    set_weight(clf, argc, argv);
#endif

#if VCTYPE == CL
    if (cmd_option_exists(argv, argv + argc, "--pierce")) {
        float p = get_cmd_option_float(argv, argv + argc, "--pierce", 0.0f);
        clf.use_piercing = true;
        clf.piercing_prob = p;
    }
#endif

    clf.prepare(argc, argv);

#if VCTYPE == CL or VCTYPE == KD
    if (clf.selftest) {
        clf.initialize_connectivity_feature();
    }
#endif

#if VCTYPE == CL
    if (cmd_option_exists(argv, argv + argc, "--hit_log") || cmd_option_exists(argv, argv + argc, "--hitlog")) {
        clf.initialize_hits_logging(out_dir + "/hit_log.compressed");
    }
#endif

    std::string loaddir = get_cmd_option_string(argv, argv + argc, "--load", "");
    if (!loaddir.empty()) {
        clf.load_classification_data(loaddir);
    }

    std::ofstream acc_log(out_dir + "/acc_log.txt");

    if (!silent)
        std::cout << "Starting iterations" << std::endl;
    timer iterations_timer;
    int niter_a = get_cmd_option_int(argv, argv + argc, "--niter_a", 1);
    int niter_b = get_cmd_option_int(argv, argv + argc, "--niter_b", 100);
    for (int i = 0; i < niter_b; i++) {
        clf.perform_iterations_and_update(niter_a);
        if (!silent)
            std::cout << (i + 1) * niter_a << ": " << clf.accuracy << std::endl;
        acc_log << (i + 1) * niter_a << " " << clf.accuracy << std::endl;
    }
    if (!silent)
        std::cout << "All iterations took " << iterations_timer.elapsed() << " sec." << std::endl;
    acc_log.close();

    clf.save_classification_data(out_dir);
}

void save_areas(int argc, char **argv) {
    assert(argc >= 2);
    auto out_dir = init_out_dir(argc, argv);

    std::string train_filename = argv[1];

    int seed = get_cmd_option_int(argv, argv + argc, "--seed", 239);
    VoronoiClassifier clf(seed);
    bool silent = false;
    if (cmd_option_exists(argv, argv + argc, "--silent")) {
        silent = true;
        clf.print_info = false;
    }
    if (!silent)
        std::cout << "Reading train data" << std::endl;
    clf.load_train_data(train_filename);
    std::fill(clf.train_labels.begin(), clf.train_labels.end(), 0);     // !!!!!!!!!
    clf.k = 1;                                                          // !!!!!!!!!
    if (cmd_option_exists(argv, argv + argc, "--selftest")) {
        if (!silent)
            std::cout << "Initializing selftest" << std::endl;
        clf.init_selftest();
    } else {
        if (!silent)
            std::cout << "Reading test data" << std::endl;
        std::string test_filename = argv[2];
        clf.load_test_data(test_filename);
    }
    if (!silent)
        std::cout << "Initializing kernels" << std::endl;
#if VCTYPE == CL or VCTYPE == KD
    set_weight(clf, argc, argv);
#endif
    clf.prepare(argc, argv);

    goto_out_dir(out_dir);

    ensure(clf.k == 1, "Should be only one class");

    if (!silent)
        std::cout << "Starting iterations" << std::endl;
    timer iterations_timer;
    int niter_a = get_cmd_option_int(argv, argv + argc, "--niter_a", 1);
    int niter_b = get_cmd_option_int(argv, argv + argc, "--niter_b", 100);
    vec2f matrix = make_vec2d(clf.test_n, niter_b);

    bool true_integral = cmd_option_exists(argv, argv + argc, "--true_integral");

    for (int i = 0; i < niter_b; i++) {
        clf.perform_iterations_and_update(niter_a);

        for (int j = 0; j < clf.test_n; j++) {
            if (true_integral) {
                matrix[j][i] = clf.series_to_integral_value(j, 0);  // multiplication by sphere volume is generally bad
            } else {
                matrix[j][i] = clf.series[j][0] / clf.samples_cnt[j];
            }
        }
        if (!silent)
            std::cout << i << std::endl;
    }
    if (!silent)
        std::cout << "All iterations took " << iterations_timer.elapsed() << " sec." << std::endl;

    save_matrix(matrix, "areas.txt");
}

#if VCTYPE == CL
void calc_dxdx(int argc, char **argv) {
    assert(argc >= 3);
    auto out_dir = init_out_dir(argc, argv);

    std::string train_filename = argv[1];
    std::string test_filename = argv[2];

    int seed = get_cmd_option_int(argv, argv + argc, "--seed", 239);
    VoronoiClassifier clf(seed);
    std::cout << "Reading train data" << std::endl;
    clf.load_train_data(train_filename);
    std::cout << "Reading test data" << std::endl;
    clf.load_test_data(test_filename);

    set_weight(clf, argc, argv);
    clf.prepare(argc, argv);

    goto_out_dir(out_dir);

    clf.save_dxdx("dxdx.npz");
}
#endif


int main(int argc, char **argv) {
    std::string task = get_cmd_option_string(argv, argv + argc, "--task", "classify");

    if (task == "classify") {
        run_classification(argc, argv);
    } else if (task == "save_areas") {
        save_areas(argc, argv);
#if VCTYPE == CL
    } else if (task == "calc_dxdx") {
        std::cout << "Starting dxdx calculation task." << std::endl;
        calc_dxdx(argc, argv);
#endif
    } else {
        std::cerr << "Unknown task: " << task << std::endl;
    }

    return 0;
}
