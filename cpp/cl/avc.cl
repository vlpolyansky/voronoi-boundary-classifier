#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <cfloat>
#include <cstdio>
#include <cmath>
#endif

#line 7


#ifndef PRECALCULATE_DXDX_WORKGROUP_SIZE
#define PRECALCULATE_DXDX_WORKGROUP_SIZE 1
#endif
#ifndef PRECALCULATE_DXDX_WORKGROUP_SIZE_GRID
#define PRECALCULATE_DXDX_WORKGROUP_SIZE_GRID 1
#endif
__kernel void precalculate_dxdx(__global const float *test_data,
                                __global const float *train_data,
                                __global       float *dxdx_data,
                                int rows, int cols, int d) {
    const unsigned int test_idx = get_global_id(1);
    const unsigned int train_idx = get_global_id(2);
    const unsigned int loc_0 = get_local_id(0);
    const unsigned int loc_1 = get_local_id(1);
    const unsigned int loc_2 = get_local_id(2);

    float sum = 0.0f;
    float t;

    if (test_idx < rows && train_idx < cols) {
        for (int i = loc_0; i < d; i += PRECALCULATE_DXDX_WORKGROUP_SIZE) {
            t = train_data[train_idx * d + i] - test_data[test_idx * d + i];
            sum += t * t;
        }
    }

    __local float sum_local[PRECALCULATE_DXDX_WORKGROUP_SIZE]
    [PRECALCULATE_DXDX_WORKGROUP_SIZE_GRID]
    [PRECALCULATE_DXDX_WORKGROUP_SIZE_GRID];
    sum_local[loc_0][loc_1][loc_2] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (test_idx < rows && train_idx < cols && loc_0 == 0) {
        sum = 0.0f;
        for (int i = 0; i < PRECALCULATE_DXDX_WORKGROUP_SIZE; i++) {
            sum += sum_local[i][loc_1][loc_2];
        }
        dxdx_data[test_idx * cols + train_idx] = sum;
    }
}


#ifndef PRECALCULATE_UR_WORKGROUP_SIZE_GRID
#define PRECALCULATE_UR_WORKGROUP_SIZE_GRID 1
#endif
#ifndef PRECALCULATE_UR_WORKGROUP_SIZE
#define PRECALCULATE_UR_WORKGROUP_SIZE 1
#endif
__kernel void precalculate_ur(__global const float *data_t,
                              __global const float *u,
                              __global       float *ur_data,
                              int rows, int d) {
    const unsigned int index = get_global_id(0);
    const unsigned int loc_0 = get_local_id(0);
    const unsigned int loc_1 = get_local_id(1);

    float sum = 0.0f;
    if (index < rows) {
        for (int i = loc_1; i < d; i += PRECALCULATE_UR_WORKGROUP_SIZE) {
            //sum += data[index * d + i] * u[i];
            sum += data_t[i * rows + index] * u[i];
        }
    }

    __local float sum_local[PRECALCULATE_UR_WORKGROUP_SIZE_GRID][PRECALCULATE_UR_WORKGROUP_SIZE];
    sum_local[loc_0][loc_1] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (index < rows && loc_1 == 0) {
        sum = 0.0f;
        for (int i = 0; i < PRECALCULATE_UR_WORKGROUP_SIZE; i++)
            sum += sum_local[loc_0][i];
        ur_data[index] = sum;
    }
}

__kernel void transpose(__global float *data, __global float *data_t, int rows, int cols)
{
    const unsigned int idx0 = get_global_id(0);
    const unsigned int idx1 = get_global_id(1);
    if (idx0 >= rows || idx1 >= cols)
        return;

    data_t[idx1 * rows + idx0] = data[idx0  * cols + idx1];
}