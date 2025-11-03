extern "C" {

void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_add<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vadd<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

} // extern "C" 