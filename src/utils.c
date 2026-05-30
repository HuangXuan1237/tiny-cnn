#include "utils.h"

#include <math.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <sys/random.h>

void utils_get_system_entropy(void *buffer, size_t size) {
    if (getrandom(buffer, size, 0) < 0) {
        FILE *file = fopen("/dev/urandom", "rb");
        
        if (file) {
            if (fread(buffer, 1, size, file) != size) {
                fclose(file);
                return;
            }

            fclose(file);
        }
    }
}

void utils_debug_grad_stats(const nn_model *model, const char *phase) {
    printf("\n[%s] Gradient Statistics:\n", phase);

    float total_grad_norm = 0.0F;
    float total_weight_norm = 0.0F;
    size_t num_params = 0;

    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        const nn_tensor *t = model->backward_graph.tensors[i];

        if ((t->flags & TENSOR_AS_PARAM) != TENSOR_AS_PARAM) {
            continue;
        }

        float grad_norm = 0.0F;
        size_t grad_size = t->grad->rows * t->grad->cols;
        for (size_t j = 0; j < grad_size; j++) {
            grad_norm += t->grad->data[j] * t->grad->data[j];
        }

        grad_norm = sqrtf(grad_norm);

        float weight_norm = 0.0F;
        size_t weight_size = t->value->rows * t->value->cols;
        for (size_t j = 0; j < weight_size; j++) {
            weight_norm += t->value->data[j] * t->value->data[j];
        }

        weight_norm = sqrtf(weight_norm);

        float ratio = (weight_norm > 1e-8F) ? grad_norm / weight_norm : 0.0F;

        printf(
            "  Tensor[%2zu] (shape %zux%zu): |grad| = %.6F, |weight| = %.6F, ratio = %.6F\n",
            t->index, t->value->rows, t->value->cols, grad_norm, weight_norm, ratio
        );

        total_grad_norm += grad_norm * grad_norm;
        total_weight_norm += weight_norm * weight_norm;
        num_params++;
    }

    total_grad_norm = sqrtf(total_grad_norm);
    total_weight_norm = sqrtf(total_weight_norm);
    float global_ratio = (total_weight_norm > 1e-8F) ? total_grad_norm / total_weight_norm : 0.0F;

    printf(
        "  [Total over %zu params] |grad| = %.6F, |weight| = %.6F, ratio = %.6F\n",
        num_params, total_grad_norm, total_weight_norm, global_ratio
    );
}

matrix *_get_image_view1(arena *ar, const matrix *src, size_t index) {
    if (src == NULL || index >= src->rows) {
        return NULL;
    }

    matrix *view = (matrix*)arena_alloc(ar, sizeof(matrix), 1);
    
    view->rows = 1;
    view->cols = src->cols; 
    
    view->data = src->data + index * src->cols;

    return view;
}

matrix *_get_image_view2(arena *ar, const dataset *src, size_t index) {
    if (src == NULL || index >= src->images->rows) {
        return NULL;
    }

    matrix *view = (matrix*)arena_alloc(ar, sizeof(matrix), 1);
    
    view->rows = 1;
    view->cols = src->images->cols; 
    
    view->data = src->images->data + index * src->images->cols;

    return view;
}

matrix *_get_image_view3(arena *ar, const dataloader *src, size_t index) {
    if (src == NULL || index >= src->images->rows) {
        return NULL;
    }

    matrix *view = (matrix*)arena_alloc(ar, sizeof(matrix), 1);
    
    view->rows = 1;
    view->cols = src->images->cols; 
    
    view->data = src->images->data + index * src->images->cols;

    return view;
}

// 💀
void utils_draw_image(const matrix *image, size_t w, size_t h, size_t c) {
    size_t image_size = w * h * c;

    if (image == NULL) {
        return;
    }
    
    uint8_t *pixels = (uint8_t*)malloc(image_size);
    const float *image_data = image->data;

    for (size_t hi = 0; hi < h; hi++) {
        for (size_t wi = 0; wi < w; wi++) {
            for (size_t ci = 0; ci < c; ci++) {
                float val = image_data[ci * (h * w) + hi * w + wi] * 255.0F;
                pixels[(hi*w+wi)*c+ci] = (uint8_t)fminf(fmaxf(val, 0.0F), 255.0F);
            }
        }
    }

    const char *temp_file = "temp.png";

    if (stbi_write_png(temp_file, (int)w, (int)h, (int)c, pixels, (int)(w*c))) {
        char cmd[256];
        const char *tmux_env = getenv("TMUX");
        if (tmux_env != NULL) {
            snprintf(
                cmd, sizeof(cmd),
                "chafa --passthrough tmux -f kitty --dither none --size 20x20 %s", 
                temp_file
            );
        } else {
            snprintf(
                cmd, sizeof(cmd),
                "chafa -f kitty --dither none --size 20x20 %s", 
                temp_file
            );
        }

        if (system(cmd) != 0) {
            printf("🤣👉🤡\n");
        }
    }

    free(pixels);
    remove(temp_file);
}