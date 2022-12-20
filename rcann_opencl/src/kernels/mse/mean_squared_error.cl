__kernel void mean_squared_error(
        const uint ROWS,
        const uint COLS,
        const uint ROW_STRIDE,
        __global realX* output,
        __global realX* expected,
        __global real* result,
        __global realX* result_deriv
) {
    const uint row = get_global_id(0);
    if (row < ROWS) {

        const uint vec_cols = COLS / VECTOR_WIDTH;
        const uint vec_cols_rem = COLS % VECTOR_WIDTH;
        const uint vec_row_stride = ROW_STRIDE / VECTOR_WIDTH;
        const uint row_offset = row * vec_row_stride;
        const uint rem_offset = row_offset + vec_cols;

        /*if (vec_cols_rem > 0) {
            const uint rem_offset = row_offset + vec_cols;
            for (uint c = vec_cols_rem; c < VECTOR_WIDTH; c++) {
                VEC_IDX(output[rem_offset], c) = 0.0;
                VEC_IDX(expected[rem_offset], c) = 0.0;
            }
            vec_cols += 1;
        }*/


        realX accum = (realX)(0.0);
        for (uint c = 0; c < vec_cols; c++) {
            const uint idx = row_offset + c;
            const realX diff = output[idx] - expected[idx];
            result_deriv[idx] = diff;
            accum += diff * diff;
        }

        for (uint c = 0; c < vec_cols_rem; c++) {
            const real diff = VEC_IDX(output[rem_offset], c) - VEC_IDX(expected[rem_offset], c);
            VEC_IDX(result_deriv[rem_offset], c) = diff;
            VEC_IDX(accum, c) += diff * diff;
        }

        result[row] = VEC_DOT_SCALAR(accum, (real)1.0/(real)COLS);
    }
}