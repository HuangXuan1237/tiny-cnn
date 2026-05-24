#pragma once

#include <stdint.h>

#include "obs.h"
#include "matrix.h"

#define RESIZE(w, h) \
    _pack_uint16_by2(w, h, _RESIZE)

#define RANDOM_HFLIP(prob) \
    _pack_float(prob, _RANDOM_HFLIP)

#define RANDOM_CROP(padding) \
    _pack_uint8(padding, _RANDOM_CROP)

#define H(x) size_t(x)

typedef enum data_source {
    MNIST,
    FashionMNIST,
    CIFAR10
} data_source;

typedef enum data_tranforms {
    NORMALIZE       =   1 << 0,
    STANDARDIZE     =   1 << 1,
    _RESIZE         =   1 << 2,
    _RANDOM_HFLIP   =   1 << 3,
    _RANDOM_CROP    =   1 << 4
} data_tranforms;

typedef struct dataset {
    obs *stk;

    matrix *images;
    matrix *labels;

    size_t size;

    size_t w, h, c; //NOSONAR
    size_t t;

    bool is_normalized;
    bool is_standardized;
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

    bool is_start;
    size_t batch_index;
} dataloader;

uint64_t _pack_uint16_by2(uint16_t value1, uint16_t value2, uint8_t flag);
uint64_t _pack_float(float value, uint8_t flag);
uint64_t _pack_uint8(uint8_t value, uint8_t flag);

dataset *dset_create(obs *stk);

bool dset_load_cifar10(dataset *train_ds, dataset *val_ds, uint64_t transforms);

size_t dset_load_image_folder(dataset *ds, const char *folder_path, uint64_t transforms);

dataset *dset_subset(obs *stk, const dataset *ds, size_t class_count, const int *selected_classes);

dataloader *dloader_create(obs *stk, const dataset *ds, size_t batch_size, bool shuffle);

void dloader_apply_label_smoothing(dataloader *loader, float smoothing, size_t num_classes);

bool dloader_iterate(dataloader *loader);

void dloader_reset(dataloader *loader);