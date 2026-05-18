#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define CIFAR10_W   32  // CIFAR-10 图像宽度
#define CIFAR10_H   32  // CIFAR-10 图像高度
#define CIFAR10_C   3   // CIFAR-10 图像通道数
#define CIFAR10_T   10  // CIFAR-10 分类数

void get_system_entropy(void *buffer, size_t size) {
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

void debug_grad_stats(const nn_model *model, const char *phase) {
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

matrix *load_image(stack *stk, const char *path, size_t w, size_t h, size_t c) {
    int width, height, channels; // NOSONAR
    uint8_t *image_data = stbi_load(path, &width, &height, &channels, 3);

    if (image_data == NULL) {
        return NULL;
    }

    matrix *image = mat_create(stk, 1, w*h*c);

    for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
            float gw = (float)wi * (float)(width - 1) / (float)(w - 1);
            float gh = (float)hi * (float)(height - 1) / (float)(h - 1);
            
            int gwi = (int)gw;
            int ghi = (int)gh;

            int gwi_next = (gwi + 1 < width) ? gwi + 1 : gwi;
            int ghi_next = (ghi + 1 < height) ? ghi + 1 : ghi;
            
            float dw = gw - (float)gwi;
            float dh = gh - (float)ghi;

            for (int ci = 0; ci < c; ci++) {
                float c00 = image_data[(ghi*width+gwi)*3+ci] / 255.0F;
                float c10 = image_data[(ghi*width+gwi_next)*3+ci] / 255.0F;
                float c01 = image_data[(ghi_next*width+gwi)*3+ci] / 255.0F;
                float c11 = image_data[(ghi_next*width+gwi_next)*3+ci] / 255.0F;

                float value = (
                    (1 - dw) * (1 - dh) * c00 +
                    dw * (1 - dh) * c10 +
                    (1 - dw) * dh * c01 +
                    dw * dh * c11
                );

                image->data[ci*(h*w)+hi*w+wi] = value;
            }
        }
    }

    stbi_image_free(image_data);

    return image;
}

void standardize_image(const matrix *result, const matrix *image, size_t w, size_t h, size_t c) {
    if (image == NULL || image->data == NULL || result == NULL || result->data == NULL) {
        return;
    }

    size_t spatial = w * h;
    size_t image_size = spatial * c;

    for (size_t b = 0; b < image->rows; b++) {
        const float *image_data = image->data + (b * image_size);
        float *result_data = result->data + (b * image_size);

        for (size_t ci = 0; ci < c; ci++) {
            const float *img_channel = image_data + (ci * spatial);
            float *res_channel = result_data + (ci * spatial);

            float sum = 0.0F;
            for (size_t i = 0; i < spatial; i++) {
                sum += img_channel[i];
            }
            float mean = sum / (float)spatial;

            float sq_diff_sum = 0.0F;
            for (size_t i = 0; i < spatial; i++) {
                float diff = img_channel[i] - mean;
                sq_diff_sum += diff * diff;
            }
            float std = sqrtf(sq_diff_sum / (float)spatial);
            float inv_std = 1.0F / (std + 1e-8F);

            for (size_t i = 0; i < spatial; i++) {
                res_channel[i] = (img_channel[i] - mean) * inv_std;
            }
        }
    }
}

void draw_image(const matrix *image, size_t w, size_t h, size_t c) {
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
        // snprintf( // NOSONAR
        //     cmd, sizeof(cmd),
        //     "chafa --dither none --size 20x20 %s",
        //     temp_file
        // ); 
        snprintf(
            cmd, sizeof(cmd),
            "chafa --passthrough tmux -f kitty "
            "--dither none --size 20x20 %s", 
            temp_file
        );

        if (system(cmd) != 0) {
            printf("🤣👉🤡\n");
        }
    }

    free(pixels);
    remove(temp_file);
}

static bool _is_image_exist(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (ext == NULL) {
        return 0;
    }

    return (
        strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0 ||
        strcasecmp(ext, ".png") == 0 || strcasecmp(ext, ".bmp") == 0
    );
}

matrix **load_imageset( // NOSONAR
    stack *stk, const char *folder_path,
    size_t w, size_t h, size_t c, size_t batch_size,
    size_t *image_count, size_t *batch_count
) {
    stack_marker scratch = stack_get_marker(&stk, 1);
    size_t ic = 0;

    DIR *dir = opendir(folder_path);
    if (dir == NULL) {
        *image_count = 0;
        *batch_count = 0;

        stack_drop_marker(scratch);

        return NULL;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (_is_image_exist(entry->d_name)) {
            ic++;
        }
    }
    closedir(dir);

    if (ic == 0) {
        *image_count = 0;
        *batch_count = 0;

        stack_drop_marker(scratch);

        return NULL;
    }

    char **file_paths = (char**)stack_alloc(scratch.stk, sizeof(char*) * ic, 1);
    size_t index = 0;

    dir = opendir(folder_path);
    if (dir) {
        while ((entry = readdir(dir)) != NULL) {
            if (_is_image_exist(entry->d_name)) {
                char *full_path = (char*)stack_alloc(scratch.stk, 512, 0);
                snprintf(full_path, 128, "%s/%s", folder_path, entry->d_name);

                file_paths[index++] = full_path;
            }
        }
        closedir(dir);
    }

    size_t bc = (ic + batch_size - 1) / batch_size;
    matrix **loader = (matrix**)stack_alloc(stk, sizeof(matrix*) * bc, 1);
    size_t image_size = w * h * c;

    index = 0;
    for (size_t b = 0; b < bc; b++) {
        size_t current_batch_size = (b == bc - 1) ? (ic - b * batch_size) : batch_size;
        loader[b] = mat_create(stk, current_batch_size, image_size);

        for (size_t i = 0; i < current_batch_size; i++) {
            const char *path = file_paths[index++];
            
            const matrix *temp = load_image(scratch.stk, path, w, h, c);
            
            if (temp != NULL) {
                memcpy(
                    loader[b]->data + (i * image_size), 
                    temp->data, 
                    image_size * sizeof(float)
                );   
            }
        }
    }

    *image_count = ic;
    *batch_count = bc;

    stack_drop_marker(scratch);

    return loader;
}