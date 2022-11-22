__kernel void mean_squared_error(
        const uint ROWS,
        const uint COLS,
        const uint ROW_STRIDE,
        const __global float* output,
        const __global float* expected,
        __global float* result,
        __global float* result_deriv
) {
    const uint row = get_global_id(0);
    if (row < ROWS) {
        const uint row_offset = row * ROW_STRIDE;
        float sum = 0.0f;
        for (int col = 0; col < COLS; col++) {
            const uint idx = row_offset + col;
            const float diff = output[idx] - expected[idx];
            result_deriv[idx] = diff;
            sum += diff * diff;
        }
        result[row] = sum / (float) COLS;
    }
}