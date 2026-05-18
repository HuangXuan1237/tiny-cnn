#pragma once

#include "stack.h"
#include "matrix.h"
#include "pcg32.h" // IWYU pragma: export

#include <math.h> // IWYU pragma: export
#include <omp.h>

#define RANDOM_HFLIP(prob) \
    _pack_float(prob, _RANDOM_HFLIP)

#define RANDOM_CROP(padding) \
    _pack_uint8(padding, _RANDOM_CROP)

typedef enum data_source {
    MNIST,
    FashionMNIST,
    CIFAR10
} data_source;

typedef enum data_tranforms {
    NORMALIZE       =   1 << 0,
    STANDARDIZE     =   1 << 1,
    _RANDOM_HFLIP   =   1 << 2,
    _RANDOM_CROP    =   1 << 3
} data_tranforms;

typedef struct dataset {
    stack *stk;

    matrix *images;
    matrix *labels;

    size_t size;

    size_t w, h, c; //NOSONAR
    size_t t;

    float hflip_prob;
    size_t crop_padding;
} dataset;

typedef struct dataloader {
    matrix *images;
    matrix *labels;

    size_t w, h, c; //NOSONAR

    bool shuffle;

    float hflip_prob;
    size_t crop_padding;

    float *crop_buffer;

    matrix *curr_input;
    matrix *curr_target;

    size_t batch_size;
    size_t batch_count;

    size_t curr_batch;
} dataloader;

uint32_t _pack_float(float f, uint8_t flag);
uint32_t _pack_uint8(uint8_t value, uint8_t flag);

dataset *dset_create(stack *stk);

bool dset_load(dataset *train_ds, dataset *val_ds, data_source src, uint32_t transfroms);

dataset *dset_subset(stack *stk, const dataset *ds, size_t class_count, const int *selected_classes);

dataloader *dloader_create(stack *stk, const dataset *ds, size_t batch_size, bool shuffle);

void dloader_apply_label_smoothing(dataloader *loader, float smoothing, size_t num_classes);

bool dloader_iterate(dataloader *loader);

void dloader_reset(dataloader *loader);