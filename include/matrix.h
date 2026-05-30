#pragma once

#include "arena.h"

//Row-Major
typedef struct {
    size_t rows;
    size_t cols;

    float *data;
} matrix;

matrix *mat_create(arena *ar, size_t rows, size_t cols);

bool mat_load(matrix *mat, size_t rows, size_t cols, const char *filename);

void mat_fill(matrix *mat, float value);

bool mat_add(matrix *result, const matrix *x, const matrix *y);

bool mat_argmax(const matrix *mat, size_t *result);

bool mat_batchnorm2d(
    matrix *output, const matrix *input, const matrix *gamma, const matrix *beta,
    matrix *running_mean, matrix *running_var,
    float eps, float momentum, bool is_train,
    matrix *x_hat, matrix *inv_std,
    size_t c, size_t spatial_size
);
bool mat_grad_batchnorm2d(
    matrix *grad_input, matrix *grad_gamma, matrix *grad_beta,
    const matrix *input, const matrix *gamma, 
    const matrix *x_hat, const matrix *inv_std,
    const matrix *grad_output,
    size_t c, size_t spatial_size
);

bool mat_conv2d(
    matrix *result, matrix *input, matrix *kernel, 
    size_t in_c, size_t in_h, size_t in_w, size_t k_size,
    size_t stride, size_t padding
);
bool mat_grad_conv2d(
    matrix *input_grad, matrix *kernel_grad,
    matrix *input, matrix *kernel, matrix *grad,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride, size_t padding
);

bool mat_cross_entropy(matrix *result, const matrix *pred, const matrix *target);
bool mat_grad_cross_entropy(
    matrix *pred_grad, matrix *target_grad,
    const matrix *pred, const matrix *target,
    const matrix *grad
);

bool mat_gavgpool2d(
    matrix *output, const matrix *input, size_t c, size_t h, size_t w
);
bool mat_grad_gavgpool2d(
    matrix *grad_input, const matrix *grad_output, size_t c, size_t h, size_t w
);

bool mat_maxpool2d(
    matrix *result, const matrix *input,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride
);
bool mat_grad_maxpool2d(
    matrix *result, const matrix *input, const matrix *grad,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride
);

bool mat_mul(
    matrix *result, const matrix *x, const matrix *y,
    float beta, bool transpose_x, bool transpose_y
);

bool mat_relu(matrix *result, const matrix *input);
bool mat_grad_relu(
    matrix *result, const matrix *input, const matrix *grad
);

bool mat_softmax(matrix *result, const matrix *input);
bool mat_grad_softmax(
    matrix *result, const matrix *input, const matrix *grad
);

float mat_sum(const matrix *mat);

bool mat_scale(matrix *mat, float scalar);