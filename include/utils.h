#pragma once

#include <dirent.h>
#include <libgen.h>
#include <stdint.h>
#include <sys/random.h>
#include <unistd.h>

#include "matrix.h"
#include "nn.h"

#define KiB(n) ( (size_t)(n) << 10 )    //2^10
#define MiB(n) ( (size_t)(n) << 20 )    //2^20
#define GiB(n) ( (size_t)(n) << 30 )    //2^30

void get_system_entropy(void *buffer, size_t size);

void project_init();

void debug_grad_stats(const nn_model *model, const char *phase);

matrix *load_image(stack *stk, const char *path, size_t w, size_t h, size_t c);

void standardize_image(const matrix *result, const matrix *image, size_t w, size_t h, size_t c);

void draw_image(const matrix *image, size_t w, size_t h, size_t c);

matrix **load_imageset(
    stack *stk, const char *folder_path,
    size_t w, size_t h, size_t c, size_t batch_size,
    size_t *image_count, size_t *batch_count
);