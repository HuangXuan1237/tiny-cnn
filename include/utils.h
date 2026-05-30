#pragma once

#include "data_utils.h"
#include "matrix.h"
#include "nn.h"

#define utils_get_image_view(ar, src, index) \
    _Generic(src, \
        matrix *: _get_image_view1, \
        dataset *: _get_image_view2, \
        dataloader *: _get_image_view3 \
    )(ar, src, index)

void utils_get_system_entropy(void *buffer, size_t size);

void log_grad_stats(const nn_model *model, const char *phase);

matrix *_get_image_view1(arena *ar, const matrix *src, size_t index);
matrix *_get_image_view2(arena *ar, const dataset *src, size_t index);
matrix *_get_image_view3(arena *ar, const dataloader *src, size_t index);

void utils_draw_image(const matrix *image, size_t w, size_t h, size_t c);