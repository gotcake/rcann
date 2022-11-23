__kernel void sigmoid(
        const __global realX* activation,
        __global realX* output
) {
    const uint offset = get_global_id(0) * VECTOR_PER_THREAD;
    #pragma unroll
    for (uint k = 0; k < VECTOR_PER_THREAD; k++) {
        output[offset + k] = (realX)(1.0) / ((realX)(1.0) + exp(-activation[offset + k]));
    }
}

__kernel void sigmoid_error(
        const __global realX* output,
        const __global realX* error,
        __global realX* result
) {
    const uint offset = get_global_id(0) * VECTOR_PER_THREAD;
    #pragma unroll
    for (uint k = 0; k < VECTOR_PER_THREAD; k++) {
        const realX out = output[offset + k];
        result[offset + k] = error[offset + k] * (out *  ((realX)(1.0) - out));
    }
}

__kernel void add_assign(
        const float alpha,
        const float beta,
        const __global realX* input,
        __global realX* output
) {
    const uint offset = get_global_id(0) * VECTOR_PER_THREAD;
    #pragma unroll
    for (uint k = 0; k < VECTOR_PER_THREAD; k++) {
        output[offset + k] = alpha * input[offset + k] + beta * output[offset + k];
    }
}

__kernel void column_sum(
        const uint ROWS,
        const uint COLS,
        const uint ROW_STRIDE,
        const float alpha,
        const float beta,
        const __global realX* input,
        __global realX* output
) {
    const uint col = get_global_id(0);
    realX sum = (realX)(0.0);
    for (uint row = 0; row < ROWS; row++) {
        sum += input[row * ROW_STRIDE + col];
    }
    output[col] = alpha * sum + beta * output[col];
}
