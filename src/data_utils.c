#include "data_utils.h"

#define MNIST_W 28
#define MNIST_H 28
#define MNIST_C 1
#define MNIST_T 10

#define CIFAR10_W 32  //CIFAR-10 图像宽度
#define CIFAR10_H 32  //CIFAR-10 图像高度
#define CIFAR10_C 3   //CIFAR-10 图像通道数
#define CIFAR10_T 10  //CIFAR-10 分类数

uint32_t _pack_float(float f, uint8_t flag) {
    f = fmaxf(fminf(f, 1), 0);
    
    uint16_t q = (uint16_t)(f * 65535.0f + 0.5f);
    
    return ((uint32_t)q << 16) | (flag & 0xFF);
}

uint32_t _pack_uint8(uint8_t value, uint8_t flag) {
    return ((uint32_t)value << 8) | flag;
}

static float __unpack_float(uint32_t packed) {
    uint16_t q = packed >> 16;

    return q / 65535.0f;
}

static uint8_t __unpack_uint8(uint32_t packed) {
    uint8_t u = (packed >> 8) & 0xFF;
    
    return u;
}

dataset *dset_subset(stack *stk, const dataset *ds, size_t num_selected, const int *selected_classes) {
    stack_marker scratch = stack_get_marker(&stk, 1);

    int *class_map = (int*)stack_alloc(scratch.stk, sizeof(int)*ds->t, 1);

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
        stack_drop_marker(scratch);
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
            new_label_row[new_cls] = 1.0f;

            new_index++;
        }
    }

    dataset *new_ds = (dataset*)stack_alloc(stk, sizeof(dataset), 1);

    new_ds->images = new_images;
    new_ds->labels = new_labels;
    new_ds->size = selected_train_rows;
    new_ds->w = ds->w;
    new_ds->h = ds->h;
    new_ds->c = ds->c;
    new_ds->t = num_selected;

    stack_drop_marker(scratch);

    return new_ds;
}

dataset *dset_create(stack *stk) {
    dataset *ds = (dataset*)stack_alloc(stk, sizeof(dataset), 1);

    ds->stk = stk;

    return ds;
}

bool dset_load(dataset *train_ds, dataset *val_ds, data_source src, uint32_t transfroms) { //NOSONAR
    uint32_t flags = transfroms & 0xFF;

    switch (src) {
        case MNIST: {
            size_t hw = MNIST_H * MNIST_W;
            size_t image_size = hw * MNIST_C;
            size_t label_size = MNIST_T;

            size_t train_ds_size = 50000;
            size_t val_ds_size = 10000;

            const char *train_images_path = "../data/mnist/train/train_images.mat";
            const char *train_labels_path = "../data/mnist/train/train_labels.mat";
            const char *val_images_path   = "../data/mnist/train/val_images.mat";
            const char *val_labels_path   = "../data/mnist/train/val_labels.mat";

            train_ds->images = mat_create(train_ds->stk, train_ds_size, image_size);
            train_ds->labels = mat_create(train_ds->stk, train_ds_size, label_size);
            mat_load(train_ds->images, train_ds_size, image_size, train_images_path);
            mat_load(train_ds->labels, train_ds_size, label_size, train_labels_path);

            train_ds->size = train_ds_size;
            train_ds->w = MNIST_W;
            train_ds->h = MNIST_H;
            train_ds->c = MNIST_C;
            train_ds->t = MNIST_T;

            val_ds->images = mat_create(val_ds->stk, val_ds_size, image_size);
            val_ds->labels = mat_create(val_ds->stk, val_ds_size, label_size);
            mat_load(val_ds->images, val_ds_size, image_size, val_images_path);
            mat_load(val_ds->labels, val_ds_size, label_size, val_labels_path);

            val_ds->size = val_ds_size;
            val_ds->w = MNIST_W;
            val_ds->h = MNIST_H;
            val_ds->c = MNIST_C;
            val_ds->t = MNIST_T;

            if (flags & NORMALIZE) {
                mat_scale(train_ds->images, 1 / 255.0f);
                mat_scale(val_ds->images, 1 / 255.0f);
            }

            if (flags & STANDARDIZE) {
                float mean[1] = { 33.31f };
                float std[1]  = { 78.57f };

                if (flags & NORMALIZE) {
                    mean[0] /= 255.0f;
                    std[0]  /= 255.0f;
                }

                for (size_t i = 0; i < train_ds_size; i++) {
                    float *image = train_ds->images->data + i * image_size;
                    for (size_t p = 0; p < hw; p++) { //NOSONAR
                        image[p] = (image[p] - mean[0]) / std[0];
                    }
                }

                for (size_t i = 0; i < val_ds_size; i++) {
                    float *image = val_ds->images->data + i * image_size;
                    for (size_t p = 0; p < hw; p++) { //NOSONAR
                        image[p] = (image[p] - mean[0]) / std[0];
                    }
                }
            }

            train_ds->hflip_prob = 0;
            train_ds->crop_padding = 0;
            val_ds->hflip_prob = 0;
            val_ds->crop_padding = 0;

            if (flags & _RANDOM_HFLIP) {
                train_ds->hflip_prob = __unpack_float(transfroms);
            }

            if (flags & _RANDOM_CROP) {
                train_ds->crop_padding = (size_t)__unpack_uint8(transfroms);
            }

            return 1;
        } break; //NOSONAR

        case FashionMNIST: {
            size_t hw = MNIST_H * MNIST_W;
            size_t image_size = hw * MNIST_C;
            size_t label_size = MNIST_T;

            size_t train_ds_size = 50000;
            size_t val_ds_size = 10000;

            const char *train_images_path = "../data/fashion-mnist/train/train_images.mat";
            const char *train_labels_path = "../data/fashion-mnist/train/train_labels.mat";
            const char *val_images_path   = "../data/fashion-mnist/train/val_images.mat";
            const char *val_labels_path   = "../data/fashion-mnist/train/val_labels.mat";

            train_ds->images = mat_create(train_ds->stk, train_ds_size, image_size);
            train_ds->labels = mat_create(train_ds->stk, train_ds_size, label_size);
            mat_load(train_ds->images, train_ds_size, image_size, train_images_path);
            mat_load(train_ds->labels, train_ds_size, label_size, train_labels_path);

            train_ds->size = train_ds_size;
            train_ds->w = MNIST_W;
            train_ds->h = MNIST_H;
            train_ds->c = MNIST_C;
            train_ds->t = MNIST_T;

            val_ds->images = mat_create(val_ds->stk, val_ds_size, image_size);
            val_ds->labels = mat_create(val_ds->stk, val_ds_size, label_size);
            mat_load(val_ds->images, val_ds_size, image_size, val_images_path);
            mat_load(val_ds->labels, val_ds_size, label_size, val_labels_path);

            val_ds->size = val_ds_size;
            val_ds->w = MNIST_W;
            val_ds->h = MNIST_H;
            val_ds->c = MNIST_C;
            val_ds->t = MNIST_T;

            if (flags & NORMALIZE) {
                mat_scale(train_ds->images, 1 / 255.0f);
                mat_scale(val_ds->images, 1 / 255.0f);
            }

            if (flags & STANDARDIZE) {
                float mean[1] = { 33.31f };
                float std[1]  = { 78.57f };

                if (flags & NORMALIZE) {
                    mean[0] /= 255.0f;
                    std[0]  /= 255.0f;
                }

                for (size_t i = 0; i < train_ds_size; i++) {
                    float *image = train_ds->images->data + i * image_size;
                    for (size_t p = 0; p < hw; p++) { //NOSONAR
                        image[p] = (image[p] - mean[0]) / std[0];
                    }
                }

                for (size_t i = 0; i < val_ds_size; i++) {
                    float *image = val_ds->images->data + i * image_size;
                    for (size_t p = 0; p < hw; p++) { //NOSONAR
                        image[p] = (image[p] - mean[0]) / std[0];
                    }
                }
            }

            train_ds->hflip_prob = 0;
            train_ds->crop_padding = 0;
            val_ds->hflip_prob = 0;
            val_ds->crop_padding = 0;

            if (flags & _RANDOM_HFLIP) {
                train_ds->hflip_prob = __unpack_float(transfroms);
            }

            if (flags & _RANDOM_CROP) {
                train_ds->crop_padding = (size_t)__unpack_uint8(transfroms);
            }

            return 1;
        } break; //NOSONAR

        case CIFAR10: {
            size_t hw = CIFAR10_H * CIFAR10_W;
            size_t image_size = hw * CIFAR10_C;
            size_t label_size = CIFAR10_T;

            size_t train_ds_size = 50000;
            size_t val_ds_size = 10000;

            const char *train_images_path = "../data/cifar10/train/train_images.mat";
            const char *train_labels_path = "../data/cifar10/train/train_labels.mat";    

            train_ds->images = mat_create(train_ds->stk, train_ds_size, image_size);
            train_ds->labels = mat_create(train_ds->stk, train_ds_size, label_size);

            mat_load(train_ds->images, train_ds_size, image_size, train_images_path);
            mat_load(train_ds->labels, train_ds_size, label_size, train_labels_path);

            stack_marker scratch = stack_get_marker(NULL, 0);
            float *temp_mem = (float*)stack_alloc(scratch.stk, sizeof(float) * image_size, 0);

            for (size_t i = 0; i < train_ds_size; i++) {
                float *row = train_ds->images->data + i * image_size;
                memcpy(temp_mem, row, sizeof(float) * image_size);

                for (size_t c = 0; c < CIFAR10_C; c++) {
                    for (size_t p = 0; p < hw; p++) { //NOSONAR
                        row[c*hw+p] = temp_mem[p*CIFAR10_C+c];
                    }
                }
            }

            train_ds->size = train_ds_size;
            train_ds->w = CIFAR10_W;
            train_ds->h = CIFAR10_H;
            train_ds->c = CIFAR10_C;
            train_ds->t = CIFAR10_T;

            const char *val_images_path = "../data/cifar10/train/val_images.mat";
            const char *val_labels_path = "../data/cifar10/train/val_labels.mat";

            val_ds->images = mat_create(val_ds->stk, val_ds_size, image_size);
            val_ds->labels = mat_create(val_ds->stk, val_ds_size, label_size);

            mat_load(val_ds->images, val_ds_size, image_size, val_images_path);
            mat_load(val_ds->labels, val_ds_size, label_size, val_labels_path);

            for (size_t i = 0; i < val_ds_size; i++) {
                float *row = val_ds->images->data + i * image_size;
                memcpy(temp_mem, row, sizeof(float) * image_size);

                for (size_t c = 0; c < CIFAR10_C; c++) {
                    for (size_t p = 0; p < hw; p++) { //NOSONAR
                        row[c*hw+p] = temp_mem[p*CIFAR10_C+c];
                    }
                }
            }

            val_ds->size = val_ds_size;

            val_ds->w = CIFAR10_W;
            val_ds->h = CIFAR10_H;
            val_ds->c = CIFAR10_C;
            val_ds->t = CIFAR10_T;

            stack_drop_marker(scratch);

            if (flags & NORMALIZE) {
                mat_scale(train_ds->images, 1 / 255.0f);
                mat_scale(val_ds->images, 1 / 255.0f);
            }

            if (flags & STANDARDIZE) {
                float mean[3] = { 125.307f, 122.961f, 113.8575f };
                float std[3] = { 51.5865f, 50.847f, 51.255f };

                if (transfroms & NORMALIZE) {
                    for (int i = 0; i < 3; i++) { //NOSONAR
                        mean[i] /= 255.0f;
                        std[i]  /= 255.0f;
                    }
                }

                for (size_t i = 0; i < train_ds_size; i++) {
                    float *image = train_ds->images->data + i * image_size;
                    for (size_t c = 0; c < CIFAR10_C; c++) { //NOSONAR
                        for (size_t p = 0; p < hw; p++) {
                            image[c*hw+p] = (image[c*hw+p] - mean[c]) / std[c];
                        }
                    }
                }

                for (size_t i = 0; i < val_ds_size; i++) {
                    float *image = val_ds->images->data + i * image_size;
                    for (size_t c = 0; c < CIFAR10_C; c++) { //NOSONAR
                        for (size_t p = 0; p < hw; p++) {
                            image[c*hw+p] = (image[c*hw+p] - mean[c]) / std[c];
                        }
                    }
                }
            }

            train_ds->hflip_prob = 0;
            train_ds->crop_padding = 0;

            val_ds->hflip_prob = 0;
            val_ds->crop_padding = 0;

            if (flags & _RANDOM_HFLIP) {
                train_ds->hflip_prob = __unpack_float(transfroms);
            }

            if (flags & _RANDOM_CROP) {
                train_ds->crop_padding = (size_t)__unpack_uint8(transfroms);
            }

            return 1;
        } break; //NOSONAR

        default: {
            return 0;
        } break; //NOSONAR
    }
}

dataloader *dloader_create(stack *stk, const dataset *ds, size_t batch_size, bool shuffle) {
    dataloader *loader = (dataloader*)stack_alloc(stk, sizeof(dataloader), 1);

    size_t ds_size = ds->size;
    size_t image_size = ds->w * ds->h * ds->c;
    size_t label_size = ds->t;

    loader->images = mat_create(stk, ds_size, image_size);
    loader->labels = mat_create(stk, ds_size, label_size);

    loader->c = ds->c;
    loader->h = ds->h;
    loader->w = ds->w;

    loader->shuffle = shuffle;

    loader->hflip_prob = ds->hflip_prob;
    loader->crop_padding = ds->crop_padding;

    loader->crop_buffer = (float*)stack_alloc(
        stk, sizeof(float)*batch_size*ds->c*ds->h*ds->w, 0
    );

    size_t *p = (size_t*)stack_alloc(stk, sizeof(size_t) * ds_size, 1);
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

        memcpy(
            loader->labels->data+label_size*i, 
            ds->labels->data+label_size*p[i],
            sizeof(float)*label_size
        );
    }

    loader->curr_input = mat_create(stk, batch_size, image_size);
    loader->curr_target = mat_create(stk, batch_size, label_size);

    loader->batch_size = batch_size;
    loader->batch_count = ds_size / batch_size;

    loader->curr_batch = 0;

    return loader;
}

void dloader_apply_label_smoothing(dataloader *loader, float smoothing, size_t num_classes) {
    float confidence = 1.0f - smoothing;
    float low_prob = smoothing / (float)num_classes;

    #pragma omp parallel for
    for (size_t r = 0; r < loader->labels->rows; r++) {
        float *data = loader->labels->data + r * loader->labels->cols;
        for (size_t c = 0; c < loader->labels->cols; c++) {
            if (data[c] > 0.99f) {
                data[c] = confidence + low_prob;
            } else {
                data[c] = low_prob;
            }
        }
    }
}

static void __random_hflip(matrix *images, size_t c, size_t h, size_t w, float prob) {
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

static void __random_crop( //NOSONAR
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
    size_t batch = loader->curr_batch;
    size_t batch_size = loader->batch_size;

    if (batch >= loader->batch_count) {
        return 0;
    }

    loader->curr_batch++;

    size_t image_size = loader->c * loader->h * loader->w;
    size_t label_size = loader->labels->cols;

    memcpy(
        loader->curr_input->data,
        loader->images->data+batch*batch_size*image_size,
        sizeof(float)*batch_size*image_size
    );

    memcpy(
        loader->curr_target->data,
        loader->labels->data+batch*batch_size*label_size,
        sizeof(float)*batch_size*label_size
    );

    if (loader->hflip_prob != 0.0f) {
        __random_hflip(
            loader->curr_input,
            loader->c, loader->h, loader->w,
            loader->hflip_prob
        );
    }

    if (loader->crop_padding != 0) {
        __random_crop(
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
        size_t img_row_size = loader->w * loader->h * loader->c;
        size_t label_row_size = loader->labels->cols;

        for (size_t i = 0; i < rows; i++) {
            size_t j = i + pcg32_random() % (rows - i);
            if (i != j) {
                float *img_i = loader->images->data + i * img_row_size;
                float *img_j = loader->images->data + j * img_row_size;
                memcpy(loader->crop_buffer, img_i, img_row_size * sizeof(float));
                memcpy(img_i, img_j, img_row_size * sizeof(float));
                memcpy(img_j, loader->crop_buffer, img_row_size * sizeof(float));

                float *label_i = loader->labels->data + i * label_row_size;
                float *label_j = loader->labels->data + j * label_row_size;
                for (size_t k = 0; k < label_row_size; k++) { //NOSONAR
                    float tmp = label_i[k];
                    label_i[k] = label_j[k];
                    label_j[k] = tmp;
                }
            }
        }
    }

    loader->curr_batch = 0;
}
