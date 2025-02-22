__kernel void mean_squared_error(
        const uint ROWS,
        const __global realX* output,
        const __global realX* expected,
        __global real* result,
        __global realX* result_deriv
) {
    const uint row = get_global_id(0);
    if (row >= ROWS) {
        return;
    }

    const uint row_offset = row * (ROW_STRIDE / VEC_WIDTH);

    realX accum = (realX)(0.0);
    for (uint c = 0; c < VEC_COLS; c++) {
        const uint idx = row_offset + c;
        const realX diff = output[idx] - expected[idx];
        result_deriv[idx] = diff;
        accum += diff * diff;
    }

    if (VEC_COLS_REM > 0) {
        const uint rem_offset = row_offset + VEC_COLS;
        #pragma unroll
        for (uint c = 0; c < VEC_COLS_REM; c++) {
            const real diff = VEC_IDX(output[rem_offset], c) - VEC_IDX(expected[rem_offset], c);
            VEC_IDX(result_deriv[rem_offset], c) = diff;
            VEC_IDX(accum, c) += diff * diff;
        }
    }

    result[row] = VEC_DOT_SCALAR(accum, (real)1.0/(real)COLS);
}