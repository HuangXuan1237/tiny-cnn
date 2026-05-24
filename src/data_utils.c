#include "data_utils.h"

#include <dirent.h>
#include <math.h>
#include <omp.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "obs.h"
#include "pcg32.h"

#define CIFAR10_W 32  
#define CIFAR10_H 32  
#define CIFAR10_C 3   
#define CIFAR10_T 10  

#define CIFAR10_T_SIZE 50000
#define CIFAR10_V_SIZE 10000

uint64_t _pack_uint16_by2(uint16_t value1, uint16_t value2, uint8_t flag) {
    uint64_t v1 = (uint64_t)value1 << 32;
    uint64_t v2 = (uint64_t)value2 << 48;
    uint64_t f = (uint64_t)flag & 0xFF;
    
    return v1 | v2 | f;
}

uint64_t _pack_float(float value, uint8_t flag) {
    value = fmaxf(fminf(value, 1), 0);
    
    uint16_t p = (uint16_t)(value * 65535.0F + 0.5F);
    
    uint64_t q = (uint64_t)p << 16;
    uint64_t f = (uint64_t)(flag & 0XFF); 
    
    return q | f;
}

uint64_t _pack_uint8(uint8_t value, uint8_t flag) {
    return ((uint64_t)value << 8) | flag;
}

static float _unpack_float(uint64_t packed) {
    uint16_t q = (uint16_t)(packed >> 16);

    return q / 65535.0F;
}

static uint8_t _unpack_uint8(uint64_t packed) {
    return (uint8_t)(packed >> 8);
}

static uint16_t _unpack_uint16_1(uint64_t packed) {
    return (uint16_t)(packed >> 32);
}

static uint16_t _unpack_uint16_2(uint64_t packed) {
    return (uint16_t)(packed >> 48);
}

dataset *dset_subset(obs *stk, const dataset *ds, size_t num_selected, const int *selected_classes) {
    obs_marker scratch = obs_get_marker(&stk, 1);

    int *class_map = (int*)obs_alloc(scratch.stk, sizeof(int)*ds->t, 1);

    for (size_t i = 0; i < ds->t; i++) {
        class_map[i] = -1; 
    }

    for (size_t i = 0; i < num_selected; i++) {
        int cls = selected_classes[i];
        
        if (cls >= 0 && cls < (int)ds->t) {
            class_map[cls] = (int)i;
        }
    }

    size_t image_size = ds->w * ds->h * ds->c;
    size_t orig_label_size = ds->t;
    size_t new_label_size = num_selected;

    size_t ds_size = ds->size;
    const matrix *images = ds->images;
    const matrix *labels = ds->labels;

    size_t selected_train_rows = 0;
    for (size_t i = 0; i < ds_size; i++) {
        const float *label_row = labels->data + i * orig_label_size;

        size_t cls = 0;
        float max_val = label_row[0];
        for (size_t j = 1; j < orig_label_size; j++) {
            if (label_row[j] > max_val) {
                max_val = label_row[j];
                cls = j;
            }
        }

        if (class_map[cls] != -1) {
            selected_train_rows++;
        }
    }

    if (selected_train_rows == 0) {
        obs_drop_marker(scratch);
        return NULL;
    }

    matrix *new_images = mat_create(stk, selected_train_rows, image_size);
    matrix *new_labels = mat_create(stk, selected_train_rows, new_label_size);

    size_t new_index = 0;
    for (size_t i = 0; i < ds_size; i++) {
        const float *label_row = labels->data + i * orig_label_size;

        size_t cls = 0;
        float max_val = label_row[0];
        for (size_t j = 1; j < orig_label_size; j++) {
            if (label_row[j] > max_val) {
                max_val = label_row[j];
                cls = j;
            }
        }
        
        int new_cls = class_map[cls];
        if (new_cls != -1) {
            memcpy(
                new_images->data+new_index*image_size,
                images->data+i*image_size,
                sizeof(float)*image_size
            );

            float *new_label_row = new_labels->data + new_index * new_label_size;
            new_label_row[new_cls] = 1.0F;

            new_index++;
        }
    }

    dataset *new_ds = (dataset*)obs_alloc(stk, sizeof(dataset), 1);

    new_ds->images = new_images;
    new_ds->labels = new_labels;
    new_ds->size = selected_train_rows;
    new_ds->w = ds->w;
    new_ds->h = ds->h;
    new_ds->c = ds->c;
    new_ds->t = num_selected;

    obs_drop_marker(scratch);

    return new_ds;
}

dataset *dset_create(obs *stk) {
    dataset *ds = (dataset*)obs_alloc(stk, sizeof(dataset), 1);

    ds->stk = stk;

    return ds;
}

bool dset_load_cifar10(dataset *train_ds, dataset *val_ds, uint64_t transforms) { // NOSONAR
    size_t hw = CIFAR10_H * CIFAR10_W;
    size_t image_size = hw * CIFAR10_C;
    size_t label_size = CIFAR10_T;

    const char *path00 = "../data/cifar10/train/train_images.mat";
    const char *path01 = "../data/cifar10/train/train_labels.mat";    

    train_ds->images = mat_create(train_ds->stk, CIFAR10_T_SIZE, image_size);
    train_ds->labels = mat_create(train_ds->stk, CIFAR10_T_SIZE, label_size);

    mat_load(train_ds->images, CIFAR10_T_SIZE, image_size, path00);
    mat_load(train_ds->labels, CIFAR10_T_SIZE, label_size, path01);

    obs_marker marker = obs_get_marker(NULL, 0);
    float *temp_mem = (float*)obs_alloc(marker.stk, sizeof(float) * image_size, 0);

    for (size_t i = 0; i < CIFAR10_T_SIZE; i++) {
        float *row = train_ds->images->data + i * image_size;
        memcpy(temp_mem, row, sizeof(float) * image_size);

        for (size_t c = 0; c < CIFAR10_C; c++) {
            for (size_t p = 0; p < hw; p++) { //NOSONAR
                row[c*hw+p] = temp_mem[p*CIFAR10_C+c];
            }
        }
    }

    train_ds->size = CIFAR10_T_SIZE;
    train_ds->w = CIFAR10_W;
    train_ds->h = CIFAR10_H;
    train_ds->c = CIFAR10_C;
    train_ds->t = CIFAR10_T;

    const char *path10 = "../data/cifar10/train/val_images.mat";
    const char *path11 = "../data/cifar10/train/val_labels.mat";

    val_ds->images = mat_create(val_ds->stk, CIFAR10_V_SIZE, image_size);
    val_ds->labels = mat_create(val_ds->stk, CIFAR10_V_SIZE, label_size);

    mat_load(val_ds->images, CIFAR10_V_SIZE, image_size, path10);
    mat_load(val_ds->labels, CIFAR10_V_SIZE, label_size, path11);

    for (size_t i = 0; i < CIFAR10_V_SIZE; i++) {
        float *row = val_ds->images->data + i * image_size;
        memcpy(temp_mem, row, sizeof(float) * image_size);

        for (size_t c = 0; c < CIFAR10_C; c++) {
            for (size_t p = 0; p < hw; p++) { //NOSONAR
                row[c*hw+p] = temp_mem[p*CIFAR10_C+c];
            }
        }
    }

    val_ds->size = CIFAR10_V_SIZE;
    val_ds->w = CIFAR10_W;
    val_ds->h = CIFAR10_H;
    val_ds->c = CIFAR10_C;
    val_ds->t = CIFAR10_T;

    obs_drop_marker(marker);

    train_ds->is_normalized = 0;
    train_ds->is_standardized = 0;
    train_ds->hflip_prob = 0;
    train_ds->crop_padding = 0;

    val_ds->is_normalized = 0;
    val_ds->is_standardized = 0;
    val_ds->hflip_prob = 0;
    val_ds->crop_padding = 0;

    uint64_t flags = transforms & 0XFF;

    if (flags & NORMALIZE) {
        train_ds->is_normalized = 1;
        val_ds->is_normalized = 1;
    }
    if (flags & STANDARDIZE) {
        train_ds->is_standardized = 1;
        val_ds->is_standardized = 1;
    }

    if (flags & _RANDOM_HFLIP) {
        train_ds->hflip_prob = _unpack_float(transforms);
    }

    if (flags & _RANDOM_CROP) {
        train_ds->crop_padding = (size_t)_unpack_uint8(transforms);
    }

    return 1;
    
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

static int _sort_paths(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b); // NOSONAR
}

matrix *_load_image(obs *stk, const char *path, size_t w, size_t h, size_t c) {
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

size_t dset_load_image_folder(dataset *ds, const char *folder_path, uint64_t transforms) { // NOSONAR
    obs_marker scratch = obs_get_marker(&ds->stk, 1);
    size_t ic = 0;

    DIR *dir = opendir(folder_path);
    if (dir == NULL) {
        obs_drop_marker(scratch);

        return 0;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (_is_image_exist(entry->d_name)) {
            ic++;
        }
    }
    closedir(dir);

    if (ic == 0) {
        obs_drop_marker(scratch);

        return 0;
    }

    char **file_paths = (char**)obs_alloc(scratch.stk, sizeof(char*) * ic, 1);
    size_t index = 0;

    dir = opendir(folder_path);
    if (dir) {
        while ((entry = readdir(dir)) != NULL) {
            if (_is_image_exist(entry->d_name)) {
                char *full_path = (char*)obs_alloc(scratch.stk, 512, 0);
                snprintf(full_path, 256, "%s/%s", folder_path, entry->d_name);

                file_paths[index++] = full_path;
            }
        }
        closedir(dir);
    }

    if (ic > 0) {
        qsort(file_paths, ic, sizeof(char *), _sort_paths);
    }

    uint64_t flags = transforms & 0XFF;

    uint16_t w = CIFAR10_W;
    uint16_t h = CIFAR10_H;
 
    if (flags & _RESIZE) {
        w = _unpack_uint16_1(transforms);
        h = _unpack_uint16_2(transforms);
    }

    size_t image_size = w * h * CIFAR10_C;

    ds->images = mat_create(ds->stk, ic, image_size);
    ds->labels = NULL;
    
    ds->size = ic;
    ds->w = w;
    ds->h = h;
    ds->c = CIFAR10_C;
    ds->t = 0;
    
    for (size_t i = 0; i < ic; i++) {
        const char *path = file_paths[i];
        const matrix *temp = _load_image(scratch.stk, path, w, h, CIFAR10_C);
        
        if (temp != NULL) {
            memcpy(
                ds->images->data + (i * image_size), 
                temp->data, 
                image_size * sizeof(float)
            );   
        }
    } 

    ds->is_normalized = flags & NORMALIZE;
    ds->is_standardized = flags & STANDARDIZE;

    ds->hflip_prob = 0;
    ds->crop_padding = 0;

    if (flags & _RANDOM_HFLIP) {
        ds->hflip_prob = _unpack_float(transforms);
    }

    if (flags & _RANDOM_CROP) {
        ds->crop_padding = (size_t)_unpack_uint8(transforms);
    }

    obs_drop_marker(scratch);
    
    // printf(
    //     "TRANSFORMS:\n"
    //     "  RESIZE: %u, %u\n"
    //     "  NORMALIZE: %s\n"
    //     "  STANDARDIZE: %s\n"
    //     "  RANDOM HFLLIP: %.2f\n"
    //     "  RANDOM CROP: %u\n",
    //     w, h,
    //     (flags & NORMALIZE) ? "TRUE" : "FALSE",
    //     (flags & STANDARDIZE) ? "TRUE" : "FALSE",
    //     ds->hflip_prob,
    //     (unsigned int)ds->crop_padding
    // );

    return ic;
}

// Spent a long time debugging this section. T_T
dataloader *dloader_create(obs *stk, const dataset *ds, size_t batch_size, bool shuffle) { // NOSONAR
    dataloader *loader = (dataloader*)obs_alloc(stk, sizeof(dataloader), 1);

    size_t ds_size = ds->size;
    size_t image_size = ds->w * ds->h * ds->c;
    size_t label_size = ds->t;

    size_t batch_count = (ds_size + batch_size - 1) / batch_size;
    size_t padded_size = batch_count * batch_size;

    loader->images = mat_create(stk, padded_size, image_size);
    loader->images->rows = ds_size; 

    loader->labels = NULL;
    if (ds->labels != NULL) {
        loader->labels = mat_create(stk, padded_size, label_size);
    }

    loader->c = ds->c;
    loader->h = ds->h;
    loader->w = ds->w;

    loader->shuffle = shuffle;

    loader->hflip_prob = ds->hflip_prob;
    loader->crop_padding = ds->crop_padding;

    loader->crop_buffer = (float*)obs_alloc(
        stk, sizeof(float)*batch_size*ds->c*ds->h*ds->w, 0
    );

    size_t *p = (size_t*)obs_alloc(stk, sizeof(size_t) * ds_size, 1);
    for (size_t i = 0; i < ds_size; i++) {
        p[i] = i;
    }

    if (shuffle) {
        for (size_t i = 0; i < ds_size; i++) {
            size_t j = i + pcg32_random() % (ds_size - i);

            size_t temp = p[i];
            p[i] = p[j];
            p[j] = temp;
        }
    }

    for (size_t i = 0; i < ds_size; i++) {
        memcpy(
            loader->images->data+image_size*i, 
            ds->images->data+image_size*p[i],
            sizeof(float)*image_size
        );

        if (ds->labels != NULL) {
            memcpy(
                loader->labels->data+label_size*i, 
                ds->labels->data+label_size*p[i],
                sizeof(float)*label_size
            );
        }
    }
    
    if (ds->is_normalized && !ds->is_standardized) {
        mat_scale(loader->images, 1/255.0F);
    }

    if (ds->is_standardized) {
        size_t spatial = loader->w * loader->h;

        for (size_t b = 0; b < loader->images->rows; b++) {
            float *image_data = loader->images->data + (b * image_size);

            for (size_t c = 0; c < loader->c; c++) {
                float *dpc = image_data + (c * spatial);

                float sum = 0.0F;
                for (size_t i = 0; i < spatial; i++) { // NOSONAR
                    sum += dpc[i];
                }
                float mean = sum / (float)spatial;

                float sq_diff_sum = 0.0F;
                for (size_t i = 0; i < spatial; i++) { // NOSONAR
                    float diff = dpc[i] - mean;
                    sq_diff_sum += diff * diff;
                }
                float std = sqrtf(sq_diff_sum / (float)spatial);
                float inv_std = 1.0F / (std + 1e-8F);

                for (size_t i = 0; i < spatial; i++) { // NOSONAR
                    dpc[i] = (dpc[i] - mean) * inv_std;
                }
            }
        }
    }

    loader->curr_input = mat_create(stk, batch_size, image_size);
    loader->curr_target = mat_create(stk, batch_size, label_size);

    loader->batch_size = batch_size;
    loader->batch_count = batch_count;

    loader->is_start = 0;
    loader->batch_index = 0;

    return loader;
}

void dloader_apply_label_smoothing(dataloader *loader, float smoothing, size_t num_classes) {
    float confidence = 1.0F - smoothing;
    float low_prob = smoothing / (float)num_classes;

    #pragma omp parallel for
    for (size_t r = 0; r < loader->labels->rows; r++) {
        float *data = loader->labels->data + r * loader->labels->cols;
        for (size_t c = 0; c < loader->labels->cols; c++) {
            if (data[c] > 0.99F) {
                data[c] = confidence + low_prob;
            } else {
                data[c] = low_prob;
            }
        }
    }
}

static void _random_hflip(matrix *images, size_t c, size_t h, size_t w, float prob) {
    size_t batch_size = images->rows;
    size_t img_size = c * h * w;
    float *data = images->data;

    uint64_t base_seed = (uint64_t)pcg32_random();

    #pragma omp parallel
    {
        pcg32_state thread_rng;
        uint64_t thread_seed = base_seed + omp_get_thread_num() * 2654435761U;
        pcg32_set_seed_r(&thread_rng, thread_seed, thread_seed);

        #pragma omp for
        for (size_t b = 0; b < batch_size; b++) {
            if (pcg32_randomf_r(&thread_rng) < prob) {
                float *img = data + b * img_size;

                for (size_t ch = 0; ch < c; ch++) {
                    float *channel = img + ch * h * w;

                    for (size_t i = 0; i < h; i++) { //NOSONAR
                        float *row = channel + i * w;

                        for (size_t j = 0; j < w / 2; j++) {
                            float tmp = row[j];
                            row[j] = row[w-1-j];
                            row[w-1-j] = tmp;
                        }
                    }
                }
            }
        }
    }
}

static void _random_crop( //NOSONAR
    matrix *images, size_t c, size_t h, size_t w,
    int padding, float *buffer
) {
    size_t batch_size = images->rows;
    size_t img_size = c * h * w;
    float *data = images->data;

    uint64_t base_seed = (uint64_t)pcg32_random();

    #pragma omp parallel
    {
        pcg32_state thread_rng;
        uint64_t thread_seed = base_seed + omp_get_thread_num() * 2654435761U;
        pcg32_set_seed_r(&thread_rng, thread_seed, thread_seed);

        #pragma omp for
        for (size_t b = 0; b < batch_size; b++) {
            int top = (pcg32_random_r(&thread_rng) % (2 * padding + 1)) - padding;
            int left = (pcg32_random_r(&thread_rng) % (2 * padding + 1)) - padding;

            float *img = data + b * img_size;
            float *cropped = buffer + b * img_size;

            for (size_t ch = 0; ch < c; ch++) {
                const float *src_channel = img + ch * h * w;
                float *dst_channel = cropped + ch * h * w;

                for (size_t i = 0; i < h; i++) {
                    int src_i = (int)i + top;
                    for (size_t j = 0; j < w; j++) { //NOSONAR
                        int src_j = (int)j + left;

                        int clamped_i = src_i;
                        int clamped_j = src_j;
                        if (clamped_i < 0) {
                            clamped_i = 0;
                        }

                        if (clamped_i >= (int)h) {
                            clamped_i = (int)h - 1;
                        }

                        if (clamped_j < 0) {
                            clamped_j = 0;
                        }

                        if (clamped_j >= (int)w) {
                            clamped_j = (int)w - 1;
                        }

                        float val = src_channel[clamped_i * w + clamped_j];
                        dst_channel[i * w + j] = val;
                    }
                }
            }

            memcpy(img, cropped, img_size * sizeof(float));
        }
    }
}

bool dloader_iterate(dataloader *loader) {
    loader->batch_index++;
    if (!loader->is_start) {
        loader->is_start = 1;
        loader->batch_index = 0;
    }

    size_t batch = loader->batch_index;
    size_t batch_size = loader->batch_size;

    if (batch >= loader->batch_count) {
        return 0;
    }

    size_t image_size = loader->c * loader->h * loader->w;
    size_t label_size = 0;
    if (loader->labels != NULL) {
        label_size = loader->labels->cols;
    }

    // --- 核心修改：无条件拷贝整块 Batch 内存 ---
    memcpy(
        loader->curr_input->data,
        loader->images->data + batch * batch_size * image_size,
        sizeof(float) * batch_size * image_size
    );

    if (loader->labels != NULL) {
        memcpy(
            loader->curr_target->data,
            loader->labels->data + batch * batch_size * label_size,
            sizeof(float) * batch_size * label_size
        );
    }
    // -------------------------------------------

    if (loader->hflip_prob != 0.0F) {
        _random_hflip(
            loader->curr_input,
            loader->c, loader->h, loader->w,
            loader->hflip_prob
        );
    }

    if (loader->crop_padding != 0) {
        _random_crop(
            loader->curr_input,
            loader->c, loader->h, loader->w,
            (int)loader->crop_padding, loader->crop_buffer
        );
    }

    return 1;
}

void dloader_reset(dataloader *loader) {
    if (loader->shuffle) {
        size_t rows = loader->images->rows;
        size_t image_rows = loader->w * loader->h * loader->c;
        size_t label_rows = 0;
        if (loader->labels != NULL) {
            label_rows = loader->labels->cols;
        }

        for (size_t i = 0; i < rows; i++) {
            size_t j = i + pcg32_random() % (rows - i);
            if (i != j) {
                float *img_i = loader->images->data + i * image_rows;
                float *img_j = loader->images->data + j * image_rows;
                memcpy(loader->crop_buffer, img_i, image_rows * sizeof(float));
                memcpy(img_i, img_j, image_rows * sizeof(float));
                memcpy(img_j, loader->crop_buffer, image_rows * sizeof(float));

                if (loader->labels != NULL) { // NOSONAR
                    float *label_i = loader->labels->data + i * label_rows;
                    float *label_j = loader->labels->data + j * label_rows;
                    for (size_t k = 0; k < label_rows; k++) { //NOSONAR
                        float tmp = label_i[k];
                        label_i[k] = label_j[k];
                        label_j[k] = tmp;
                    }
                }
            }
        }
    }

    loader->is_start = 0;
    loader->batch_index = 0;
}
