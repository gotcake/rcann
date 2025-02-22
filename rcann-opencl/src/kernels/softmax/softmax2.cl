/**
 *
 * @param COLS the number of columns, must be > 0
 * @param ROWS
 * @param realX
 * @return
 */
__kernel void softmax(
    const uint ROWS,
    const __global realX* activation,
    __global realX* output
) {
    const uint row = get_global_id(0);
    if (row >= ROWS) {
        return;
    }
    const uint r_vec_offset = row * ROW_STRIDE / VEC_WIDTH;

    // compute max
    realX temp_v = activation[r_vec_offset];
    //#pragma unroll
    for (uint i = 1; i < VEC_COLS; i++) {
        temp_v = fmax(temp_v, activation[r_vec_offset + i]);
    }
    real max_act = VEC_MAX(temp_v);
    temp_v = activation[r_vec_offset + VEC_COLS];
    #pragma unroll
    for (uint i = 0; i < VEC_COLS_REM; i++) {
        max_act = fmax(max_act, VEC_IDX(temp_v, i));
    }
    const realX max_act_v = (realX)max_act;

    // accumulate sum
    real sum = 0;
    //#pragma unroll
    for (uint i = 0; i < VEC_COLS; i++) {
        temp_v = exp(activation[r_vec_offset + i] - max_act_v);
        output[r_vec_offset + i] = temp_v;
        sum += VEC_DOT_SCALAR(temp_v, 1.0);
    }
    temp_v = activation[r_vec_offset + VEC_COLS];
    real out;
    #pragma unroll
    for (uint i = 0; i < VEC_COLS_REM; i++) {
        out = exp(VEC_IDX(temp_v, i) - max_act);
        VEC_IDX(output[r_vec_offset + VEC_COLS], i) = out;
        sum += out;
    }

    // divide by sum
    //#pragma unroll
    for (uint i = 0; i < VEC_COLS; i++) {
        output[r_vec_offset + i] /= sum;
    }
    if (VEC_COLS_REM > 0) {
        output[r_vec_offset + VEC_COLS] /= sum;
    }
    /*#pragma unroll
    for (uint i = 0; i < VEC_COLS_REM; i++) {
        VEC_IDX(output[r_vec_offset + VEC_COLS], i) /= sum;
    }*/

}