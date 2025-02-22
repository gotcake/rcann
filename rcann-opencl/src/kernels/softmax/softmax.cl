/**
 *
 * @param COLS the number of columns, must be > 0
 * @param ROWS
 * @param realX
 * @return
 */
__kernel void softmax(
    const uint COLS,
    const uint ROWS,
    const uint ROW_STRIDE,
    const __global realX* activation,
    __global realX* output
) {
    const uint row = get_global_id(0);
    if (row >= ROWS) {
        return;
    }
    const uint r_vec_offset = row * ROW_STRIDE / VEC_WIDTH;
    const uint vec_cols = COLS / VEC_WIDTH;
    const uint vec_cols_rem = COLS % VEC_WIDTH;

    // compute max
    realX temp_v = activation[r_vec_offset];
    for (uint i = 1; i < vec_cols; i++) {
        temp_v = fmax(temp_v, activation[r_vec_offset + i]);
    }
    real max_act = VEC_MAX(temp_v);
    temp_v = activation[r_vec_offset + vec_cols];
    for (uint i = 0; i < vec_cols_rem; i++) {
        max_act = fmax(max_act, VEC_IDX(temp_v, i));
    }
    const realX max_act_v = (realX)max_act;

    // accumulate sum
    real sum = 0;
    for (uint i = 0; i < vec_cols; i++) {
        temp_v = exp(activation[r_vec_offset + i] - max_act_v);
        output[r_vec_offset + i] = temp_v;
        sum += VEC_DOT_SCALAR(temp_v, 1.0);
    }
    temp_v = activation[r_vec_offset + vec_cols];
    for (uint i = 0; i < vec_cols_rem; i++) {
        const real out = exp(VEC_IDX(temp_v, i) - max_act);
        VEC_IDX(output[r_vec_offset + vec_cols], i) = out;
        sum += out;
    }

    // divide by sum
    for (uint i = 0; i < vec_cols; i++) {
        output[r_vec_offset + i] /= sum;
    }
    for (uint i = 0; i < vec_cols_rem; i++) {
        VEC_IDX(output[r_vec_offset + vec_cols], i) /= sum;
    }

}