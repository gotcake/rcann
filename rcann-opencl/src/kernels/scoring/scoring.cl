__kernel void compute_confusion_matrix_indices(
        const uint ROWS,
        const uint CLASSES,
        const uint OUTPUT_ROW_STRIDE,
        const uint EXPECTED_ROW_STRIDE,
        const __global real* output,
        const __global real* expected,
        __global uint2* index_buffer
) {
    const uint row = get_global_id(0);
    if (row >= ROWS) {
        return;
    }

    const uint output_offset = row * OUTPUT_ROW_STRIDE;
    uint max_output_idx = 0;
    real max_output = output[output_offset];
    for (uint i = 1; i < CLASSES; i++) {
        if (output[output_offset + i] > max_output) {
            max_output = output[output_offset + i];
            max_output_idx = i;
        }
    }

    const uint expected_offset = row * EXPECTED_ROW_STRIDE;
    uint max_expected_idx = 0;
    real max_expected = expected[expected_offset];
    for (uint i = 1; i < CLASSES; i++) {
        if (expected[expected_offset + i] > max_expected) {
            max_expected = expected[expected_offset + i];
            max_expected_idx = i;
        }
    }

    index_buffer[row] = (uint2)(max_expected_idx, max_output_idx);

}

__kernel void inc_by_indices(
        const uint NUM_INDICES,
        const uint CLASSES,
        const uint MATRIX_ROW_STRIDE,
        const __global uint2* index_buffer,
        __global real* matrix
) {
    const uint class = get_global_id(0);
    if (class >= CLASSES) {
        return;
    }
    const uint matrix_row_offset = class * MATRIX_ROW_STRIDE;
    for (uint i = 0; i < NUM_INDICES; i++) {
        uint2 index = index_buffer[i];
        if (index.x == class) {
            matrix[matrix_row_offset + index.y] += 1;
        }
    }
}