#include <set>
#include <unordered_map>
#include <unordered_map>

#include <omp.h>
#include <cnpy.h>

#include "../utils.h"
#include "../geometry.h"
#include "../range.h"

int main(int argc, char **argv) {
    ensure(argc >= 3, "Needs at least 2 arguments: graph.npy and edges.npy");
    std::string data_filename = argv[1];
    std::string edges_filename = argv[2];

    std::string distances_filename = get_cmd_option_string(argv, argv + argc, "--out", "distances.npy");
    bool is_sparse = cmd_option_exists(argv, argv + argc, "--sparse");

    ensure(!is_sparse, "Sparse output no supported [anymore...]");

    int max_neighbors = get_cmd_option_int(argv, argv + argc, "--nneigh", -1);

    ensure(max_neighbors < 0, "Max dist not fully supported [yet...]");

    cnpy::NpyArray data_npy = cnpy::npy_load(data_filename);
    ensure(data_npy.shape.size() == 2, "Data should be represented as a matrix.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32 bit.");

    int n_points = data_npy.shape[0];
    int data_dim = data_npy.shape[1];

    vec<float> data = data_npy.as_vec<float>();


    cnpy::NpyArray edges_npy = cnpy::npy_load(edges_filename);
    ensure(edges_npy.shape.size() == 2, "Edges should be represented as a matrix.");
    ensure(edges_npy.word_size == sizeof(int), "Edges word size should be 32 bit.");
    ensure(edges_npy.shape[1] >= 2, "Edges should be of size (M, 2)");
    if(edges_npy.shape[1] > 2) {
        std::cout << "Warning: using only columns 0 and 1 of edges matrix" << std::endl;
    }
    int n_edges = edges_npy.shape[0];
    int edge_width = edges_npy.shape[1];

    vec<int> edges_list = edges_npy.as_vec<int>();

    vec<float> weights(n_edges);
    vec<vec<std::pair<int, float>>> edges(n_points);

    std::cout << "Computing edge weights" << std::endl;
    my_tqdm bar(n_edges);


    #pragma omp parallel for
    for (int i = 0; i < n_edges; i++) {
        bar.atomic_iteration();
        int u = edges_list[i * edge_width];
        int v = edges_list[i * edge_width + 1];
        float w = length(range<float>(data.begin() + u * data_dim, data.begin() + (u + 1) * data_dim) -
                range<float>(data.begin() + v * data_dim, data.begin() + (v + 1) * data_dim));
        weights[i] = w;
    }
    for (int i = 0; i < n_edges; i++) {
        int u = edges_list[i * edge_width];
        int v = edges_list[i * edge_width + 1];
        float w = weights[i];
        edges[u].push_back(std::make_pair(v, w));
        edges[v].push_back(std::make_pair(u, w));
    }
    bar.bar().finish();
//
//    vec<std::unordered_map<int, std::pair<float, bool>>> distances(n_points);
    vec<vec<float>> distances(n_points);

    std::cout << "Computing shortest distances" << std::endl;
    bar = my_tqdm(n_points);
    
    #pragma omp parallel for
    for (int s = 0; s < n_points; s++) {
        bar.atomic_iteration();

//        std::unordered_map<int, std::pair<float, bool>> distances_to;   // (dist, is_final)
        vec<float> distances_to(n_points, -1);
        distances_to[s] = 0;

        std::set<std::pair<float, int>> q;
        q.insert(std::make_pair(.0f, s));
        for (int it = 0; !q.empty() && (max_neighbors == -1 || it < max_neighbors + 1); it++) {
            int u = q.begin()->second;
            float d_u = distances_to[u];
//            float d_u = distances_to.find(u)->second.first;
            q.erase(q.begin());

            for (const std::pair<int, float> &p: edges[u]) {
                int v = p.first;
                float w = p.second;

                float d_v = distances_to[v];


                if (d_v < 0 || d_v > d_u + w) {
                    q.erase(std::make_pair(d_v, v));
                    distances_to[v] = d_u + w;
                    // can add parent info here
                    q.insert(std::make_pair(d_u + w, v));
                }
            }
        }

        distances[s] = distances_to;
    }
    bar.bar().finish();

    std::cout << "Saving results to " << distances_filename << std::endl;
    if (is_sparse) {
//        vec<float> output;
//        for (int u = 0; u < n_points; u++) {
//            for (const auto& p: distances[u]) {
//                if (p.second.second) {
//                    output.push_back(float(u));
//                    output.push_back(float(p.first));
//                    output.push_back(p.second.first);
//                }
//            }
//        }
//        cnpy::npy_save(distances_filename, output.data(), {output.size() / 3, 3});
    } else {
        vec<float> output(n_points * n_points, -1);
        for (int u = 0; u < n_points; u++) {
            for (int v = 0; v < n_points; v++) {
                output[u * n_points + v] = distances[u][v];
            }
        }
        cnpy::npy_save(distances_filename, output.data(), {size_t(n_points), size_t(n_points)});
    }


    return 0;
}