#pragma once

#include <stdint.h>

#include "arena.h"
#include "matrix.h"

#define NN_MAX_INPUT_COUNT 3

typedef enum {
    TENSOR_DEFAULT          =   0,
    TENSOR_REQUIRES_GRAD    = 	1 << 0,
    TENSOR_AS_PARAM         = 	1 << 1,
    TENSOR_AS_INPUT         = 	1 << 2,
    TENSOR_AS_OUTPUT        = 	1 << 3,
    TENSOR_AS_TARGET        = 	1 << 4,
    TENSOR_AS_LOSS          = 	1 << 5,
} nn_tensor_flags;

typedef enum {
    TENSOR_OP_NONE,

    _TENSOR_OP_UNARY,
    TENSOR_OP_DROPOUT,
    TENSOR_OP_MAXPOOL2D,
    TENSOR_OP_GAPOOL2D,
    TENSOR_OP_RELU,
    TENSOR_OP_SOFTMAX,

    _TENSOR_OP_BINARY,
    TENSOR_OP_ADD,
    TENSOR_OP_CONV2D,
    TENSOR_OP_CROSS_ENTROPY,
    TENSOR_OP_MATMUL,

    TENSOR_OP_BATCHNORM2D
} nn_tensor_ops;

typedef struct {
    size_t index;
    uint32_t flags;

    size_t c, h, w; // NOSONAR

    size_t stride;
    size_t padding;
    size_t k_size;

    float eps;
    float momentum;
    matrix *running_mean;
    matrix *running_var;
    matrix *aux;
    matrix *aux2;

    matrix *value;
    matrix *grad;
    
    nn_tensor_ops op;

    struct nn_tensor *inputs[NN_MAX_INPUT_COUNT];    
} nn_tensor;

typedef struct {
    nn_tensor **tensors;
    size_t tensor_count;
} _nn_graph;

typedef struct {
    size_t tensor_count;
    
    nn_tensor *input;
    nn_tensor *output;
    nn_tensor *target;
    nn_tensor *loss;

    size_t *classes;

    _nn_graph forward_graph;
    _nn_graph backward_graph;
} nn_model;

nn_tensor *nn_tensor_create(
    arena *ar, nn_model *model,
    size_t c, size_t h, size_t w,
    size_t batch_size, uint32_t flags
);

nn_tensor *nn_func_dropout(
    arena *ar, nn_model *model,
    nn_tensor *input, float p, uint32_t flags
);

nn_tensor *nn_func_maxpool2d(
    arena *ar, nn_model *model,
    nn_tensor *input, size_t k_size, size_t stride,
    uint32_t flags
);

nn_tensor *nn_func_gapool2d(
    arena *ar, nn_model *model,
    nn_tensor *input, uint32_t flags
);

nn_tensor *nn_func_relu(
    arena *ar, nn_model *model,
    nn_tensor *input, uint32_t flags
);

nn_tensor *nn_func_softmax(
    arena *ar, nn_model *model,
    nn_tensor *input, uint32_t flags
);

nn_tensor *nn_func_add(
    arena *ar, nn_model *model,
    nn_tensor *x, nn_tensor *y, uint32_t flags
);

nn_tensor *nn_func_conv2d(
    arena *ar, nn_model *model,
    nn_tensor *input, nn_tensor *kernel, 
    size_t k_size, size_t stride, size_t padding,
    uint32_t flags
);

nn_tensor *nn_func_cross_entropy(
    arena *ar, nn_model *model,
    nn_tensor *pred, nn_tensor *target, uint32_t flags
);

nn_tensor *nn_func_matmul(
    arena *ar, nn_model *model,
    nn_tensor *x, nn_tensor *y, uint32_t flags
);

nn_tensor *nn_func_batchnorm2d(
    arena *ar, nn_model *model,
    nn_tensor *input, nn_tensor *gamma, nn_tensor *beta,
    float eps, float momentum, uint32_t flags
);

nn_tensor *nn_layer_input(
    arena* ar, nn_model* model,
    size_t c, size_t h, size_t w,
    size_t batch_size
);
nn_tensor *nn_layer_target(
    arena* ar, nn_model* model,
    size_t c, size_t h, size_t w,
    size_t batch_size
);

nn_tensor *nn_layer_batchnorm2d(
    arena *ar, nn_model *model, nn_tensor *input
);

nn_tensor *nn_layer_conv2d(
    arena *ar, nn_model *model,
    nn_tensor *input, 
    size_t out_channels, size_t k_size,
    size_t stride, size_t padding
);

nn_tensor *nn_layer_cross_entropy(
    arena *ar, nn_model *model, nn_tensor *pred, nn_tensor *target
);

nn_tensor *nn_layer_dropout(
    arena *ar, nn_model *model, nn_tensor *input, float prob
);

nn_tensor *nn_layer_gapool2d(
    arena *ar, nn_model *model, nn_tensor *input
);

nn_tensor *nn_layer_linear(
    arena *ar, nn_model *model, nn_tensor *input, size_t out_features
);

nn_tensor *nn_layer_maxpool2d(
    arena *ar, nn_model * model, nn_tensor *input,
    size_t k_size, size_t stride
);

nn_tensor *nn_layer_relu(arena *ar, nn_model *model, nn_tensor *input);

nn_tensor *nn_layer_residual_block(
    arena *ar, nn_model *model, nn_tensor *input,
    size_t in_planes, size_t out_planes, size_t stride
);

nn_tensor *nn_layer_softmax(arena *ar, nn_model *model, nn_tensor *input);

nn_model *nn_model_create(arena *ar);
void nn_model_compile(arena *ar, nn_model *model);

float nn_model_criterion(nn_model *model, matrix *input, matrix *target);
matrix *nn_model_predict(arena *ar, nn_model *model, matrix *input);

bool nn_model_save(const nn_model *model, const char *filename);
bool nn_model_load(const nn_model *model, const char *filename);