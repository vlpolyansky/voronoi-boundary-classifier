#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <cfloat>
#include <cstdio>
#include <cmath>
#endif

#line 7


//#define WORKGROUP_SIZE 128
//__kernel void shoot_vector(__global const float *dxdx_data_t,
//                           __global const float *test_ur_data,
//                           __global const float *train_ur_data,
//                           __global       int   *best_i,
//                           __global       float *best_t,
//#ifdef SELFTEST
//        int rows, int cols, int selftest_shift) {
//#else
//                           int rows, int cols) {
//#endif
//
//    const unsigned int test_idx = get_global_id(0);
//    const unsigned int loc_idx = get_local_id(0);
//
//    __local float train_ur_data_local[WORKGROUP_SIZE];
//
//    int best_i_local = -1;
//    float best_t_local = FLT_MAX;
//
//    float test_ur_data_loaded = 0.0f;
//    if (test_idx < rows) {
//        test_ur_data_loaded = test_ur_data[test_idx];
//    }
//
//    for (int train_idx_offset = 0; train_idx_offset < cols; train_idx_offset += WORKGROUP_SIZE) {
//        if (train_idx_offset + loc_idx < cols) {
//            train_ur_data_local[loc_idx] = train_ur_data[train_idx_offset + loc_idx];
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//
//        if (test_idx < rows) {
//
//            for (int train_idx = train_idx_offset; train_idx < cols && train_idx < train_idx_offset + WORKGROUP_SIZE; train_idx++) {
//#ifdef SELFTEST
//                if (test_idx + selftest_shift == train_idx) {
//                    continue;
//                }
//#endif
//                float denom = 2 * (train_ur_data_local[train_idx - train_idx_offset] - test_ur_data_loaded);
//                float cur_t = dxdx_data_t[train_idx * rows + test_idx] / denom;
//
//                if (!isnan(cur_t) && !isinf(cur_t) && cur_t >= 0 && (best_i_local < 0 || cur_t < best_t_local)) {
//                    best_i_local = train_idx;
//                    best_t_local = cur_t;
//                }
//            }
//
//        }
//
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    if (test_idx < rows) {
//        best_i[test_idx] = best_i_local;
//        best_t[test_idx] = best_t_local;
//    }
//
//}


#define WORKGROUP_SIZE 128
#define MAX_PIERCE 5
__kernel void shoot_piercing_vector(__global const float *dxdx_data_t,
                           __global const float *test_ur_data,
                           __global const float *train_ur_data,
                                          int    pierce_n,
                           __global       int   *best_i,
                           __global       float *best_t,
#ifdef SELFTEST
        int rows, int cols, int selftest_shift) {
#else
                           int rows, int cols) {
#endif

    const unsigned int test_idx = get_global_id(0);
    const unsigned int loc_idx = get_local_id(0);

    __local float train_ur_data_local[WORKGROUP_SIZE];

    int best_i_local[MAX_PIERCE + 1];
    float best_t_local[MAX_PIERCE + 1];
    for (int i = 0; i <= pierce_n; i++) {
        best_i_local[i] = -1;
        best_t_local[i] = FLT_MAX;
    }

    float test_ur_data_loaded = 0.0f;
    if (test_idx < rows) {
        test_ur_data_loaded = test_ur_data[test_idx];
    }

    for (int train_idx_offset = 0; train_idx_offset < cols; train_idx_offset += WORKGROUP_SIZE) {
        if (train_idx_offset + loc_idx < cols) {
            train_ur_data_local[loc_idx] = train_ur_data[train_idx_offset + loc_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (test_idx < rows) {

            for (int train_idx = train_idx_offset; train_idx < cols && train_idx < train_idx_offset + WORKGROUP_SIZE; train_idx++) {
#ifdef SELFTEST
                if (test_idx + selftest_shift == train_idx) {
                    continue;
                }
#endif
                float denom = 2 * (train_ur_data_local[train_idx - train_idx_offset] - test_ur_data_loaded);
                float cur_t = dxdx_data_t[train_idx * rows + test_idx] / denom;

                if (isnan(cur_t) || isinf(cur_t) || cur_t < 0) {
                    continue;
                }
                int insert_idx = -1;
                for (int j = 0; j <= pierce_n; j++) {
                    if (best_i_local[j] < 0 || cur_t < best_t_local[j]) {
                        insert_idx = j;
                        break;
                    }
                }
                if (insert_idx >= 0) {
                    for (int j = pierce_n; j > insert_idx; j--) {
                        best_i_local[j] = best_i_local[j - 1];
                        best_t_local[j] = best_t_local[j - 1];
                    }
                    best_i_local[insert_idx] = train_idx;
                    best_t_local[insert_idx] = cur_t;
                }
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (test_idx < rows) {
        best_i[test_idx] = best_i_local[pierce_n];
        best_t[test_idx] = best_t_local[pierce_n];
    }

}