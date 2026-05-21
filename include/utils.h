#pragma once

#include <dirent.h>
#include <libgen.h>
#include <stdint.h>
#include <sys/random.h>
#include <unistd.h>

#include "data_utils.h"
#include "matrix.h"
#include "nn.h"

#define get_image_view(stk, src, index) \
    _Generic(src, \
        matrix *: _get_image_view1, \
        dataset *: _get_image_view2, \
        dataloader *: _get_image_view3 \
    )(stk, src, index)

void get_system_entropy(void *buffer, size_t size);

void log_grad_stats(const nn_model *model, const char *phase);

matrix *_get_image_view1(stack *stk, const matrix *src, size_t index);
matrix *_get_image_view2(stack *stk, const dataset *src, size_t index);
matrix *_get_image_view3(stack *stk, const dataloader *src, size_t index);

void draw_image(const matrix *image, size_t w, size_t h, size_t c);