__kernel void sigmoid(
        const __global float* activation,
        __global float* output
) {
    const uint offset = get_global_id(0) * PER_THREAD;
    #pragma unroll
    for (uint k = 0; k < PER_THREAD; k++) {
        output[offset + k] = 1.0f / (1.0f + exp(-activation[offset + k]));
    }
}

__kernel void sigmoid_error(
        const __global float* output,
        const __global float* error,
        __global float* result
) {
    const uint offset = get_global_id(0) * PER_THREAD;
    #pragma unroll
    for (uint k = 0; k < PER_THREAD; k++) {
        const float out = output[offset + k];
        result[offset + k] = error[offset + k] * (out *  (1.0f - out));
    }
}

__kernel void add_assign(
        const float alpha,
        const float beta,
        const __global float* input,
        __global float* output
) {
    const uint offset = get_global_id(0) * PER_THREAD;
    #pragma unroll
    for (uint k = 0; k < PER_THREAD; k++) {
        output[offset + k] = alpha * input[offset + k] + beta * output[offset + k];
    }
}

// presumably has worse performance
__kernel void column_sum_checked(
        const uint ROWS,
        const uint COLS,
        const uint ROW_STRIDE,
        const float alpha,
        const float beta,
        const __global float* input,
        __global float* output
) {
    const uint col = get_global_id(0);
    if (col < COLS) {
        float sum = 0.0f;
        for (uint row = 0; row < ROWS; row++) {
            sum += input[row * ROW_STRIDE + col];
        }
        output[col] = alpha * sum + beta * output[col];
    }
}

__kernel void column_sum(
        const uint ROWS,
        const uint COLS,
        const uint ROW_STRIDE,
        const float alpha,
        const float beta,
        const __global float* input,
        __global float* output
) {
    const uint offset = get_global_id(0) * PER_THREAD;

    float buff[PER_THREAD];
    #pragma unroll
    for (uint k = 0; k < PER_THREAD; k++) {
        buff[k] = 0.0f;
    }

    for (uint row = 0; row < ROWS; row++) {
        const uint row_offset = row * ROW_STRIDE + offset;
        #pragma unroll
        for (uint k = 0; k < PER_THREAD; k++) {
            buff[k] += input[row_offset + k];
        }
    }

    #pragma unroll
    for (uint k = 0; k < PER_THREAD; k++) {
        buff[k] = alpha * buff[k] + beta * output[offset + k];
    }

    #pragma unroll
    for (uint k = 0; k < PER_THREAD; k++) {
        output[offset + k] = buff[k];
    }

}