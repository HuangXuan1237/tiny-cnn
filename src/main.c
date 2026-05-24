#include <cblas.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "obs.h"
#include "data_utils.h"
#include "matrix.h"
#include "nn.h"
#include "optimizer.h"
#include "pcg32.h"
#include "utils.h"

#define NUM_SELECTED 5
#define BATCH_SIZE 64
#define EPOCHS 50
#define LEARNING_RATE 5E-4F

#define PRE_TRAINED 0
// #define MODEL_PATH "../model/current_best.bin"
#define MODEL_PATH "../model/test.bin"

const char *CIFAR10_LABELS[10] = {
    "airplane", 
    "automobile",
    "bird",
    "cat",      
    "🦌",
    "dog",
    "frog",
    "horse",    
    "ship",
    "truck"
};

void cifar10_custom_resnet(obs *stk, nn_model *model) {
    nn_tensor *input = nn_layer_input(stk, model, 3, 32, 32, BATCH_SIZE);
    nn_tensor *target = nn_layer_target(stk, model, 1, 1, 10, BATCH_SIZE);

    nn_tensor *conv1 = nn_layer_conv2d(stk, model, input, 16, 3, 1, 1);
    nn_tensor *bn1 = nn_layer_batchnorm2d(stk, model, conv1);
    nn_tensor *relu1 = nn_layer_relu(stk, model, bn1);

    nn_tensor *block1_1 = nn_layer_residual_block(stk, model, relu1, 16, 16, 1);
    nn_tensor *block1_2 = nn_layer_residual_block(stk, model, block1_1, 16, 16, 1);
    
    nn_tensor *block2_1 = nn_layer_residual_block(stk, model, block1_2, 16, 32, 2);
    nn_tensor *block2_2 = nn_layer_residual_block(stk, model, block2_1, 32, 32, 1);

    nn_tensor *block3_1 = nn_layer_residual_block(stk, model, block2_2, 32, 64, 2);
    nn_tensor *block3_2 = nn_layer_residual_block(stk, model, block3_1, 64, 64, 1);

    nn_tensor *gap = nn_layer_gapool2d(stk, model, block3_2);
    nn_tensor *dropout = nn_layer_dropout(stk, model, gap, 0.2F);
    nn_tensor *fc = nn_layer_linear(stk, model, dropout, 10);

    nn_tensor *output = nn_layer_softmax(stk, model, fc);
    nn_tensor *loss = nn_layer_cross_entropy(stk, model, output, target); // NOSONAR
}

int main() { //NOSONAR
    uint64_t seeds[2];
    utils_get_system_entropy(seeds, sizeof(seeds));

    pcg32_set_seed(seeds[0], seeds[1]);

    openblas_set_num_threads(1); // Hand over all the threading to OpenMP.
    omp_set_num_threads(8); 

    obs *global_stack = obs_create();

    dataset *the_dataset[2];
    the_dataset[0] = dset_create(global_stack);
    the_dataset[1] = dset_create(global_stack);

    dset_load_cifar10(
        the_dataset[0], the_dataset[1],
        NORMALIZE|STANDARDIZE|
        RANDOM_HFLIP(0.5)|RANDOM_CROP(4)
    );
 
    dataloader *loader[2];
    loader[0] = dloader_create(global_stack, the_dataset[0], BATCH_SIZE, 1);
    loader[1] = dloader_create(global_stack, the_dataset[1], BATCH_SIZE, 0);

    dloader_apply_label_smoothing(loader[0], 0.1F, 10);

    nn_model *model = nn_model_create(global_stack);
    
    cifar10_custom_resnet(global_stack, model);
    nn_model_compile(global_stack, model);

    // -----------------------------------TRAINING----------------------------------- //
    if (!PRE_TRAINED) {
        adam_optimizer *opt = adam_create(global_stack, model, LEARNING_RATE, 1E-4F);

        ema_optimizer *ema = ema_create(global_stack, model, 0.999F); 

        matrix **backup_space = ema_create_backup_space(global_stack, model);

        coslr_scheduler *scheduler = coslr_create(global_stack, LEARNING_RATE, 1E-5F, EPOCHS);    

        struct timespec start, end; // NOSONAR
        
        size_t *preds = obs_alloc(global_stack, sizeof(size_t) * BATCH_SIZE, 1);
        size_t *targets = obs_alloc(global_stack, sizeof(size_t) * BATCH_SIZE, 1);

        for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
            clock_gettime(CLOCK_MONOTONIC, &start);

            float epoch_loss = 0;

            while (dloader_iterate(loader[0])) {
                opt_zero_grad(model);

                float batch_loss = nn_model_criterion(
                    model, loader[0]->curr_input, loader[0]->curr_target
                ) / (float)BATCH_SIZE;

                adam_step(opt, model, BATCH_SIZE);
            
                ema_update(ema, model);

                epoch_loss += batch_loss;

                if (loader[0]->batch_index % 10 == 0) { // NOSONAR
                    printf(
                        "EPOCH %2zd/%2d | BATCH %3zd/%3zd | LOSS: %.4f\r",
                        epoch + 1, EPOCHS, loader[0]->batch_index, loader[0]->batch_count, batch_loss
                    );
                    fflush(stdout);
                }
            }

            dloader_reset(loader[0]);

            // -----------------------------------VALIDATION----------------------------------- //
            size_t num_matched = 0;
            size_t sc = 0;
            
            ema_apply_shadow(ema, model, backup_space);

            while (dloader_iterate(loader[1])) {
                const matrix *pred = nn_model_predict(global_stack, model, loader[1]->curr_input);

                mat_argmax(pred, preds);
                mat_argmax(loader[1]->curr_target, targets);

                for (size_t i = 0; i < BATCH_SIZE; i++) { // NOSONAR
                    if (preds[i] == targets[i]) {
                        num_matched++;
                    }
                }

                sc += BATCH_SIZE;
            }

            dloader_reset(loader[1]);

            opt_coslr_step(scheduler, opt);

            ema_restore_weight(ema, model, backup_space);

            clock_gettime(CLOCK_MONOTONIC, &end);

            double elapsed = (
                (double)(end.tv_sec - start.tv_sec) +
                (double)(end.tv_nsec - start.tv_nsec) / 1E9
            );

            printf(
                "\rEPOCH %2zd | AVG LOSS: %.4f | 验证准确率: %.2f%% | 耗时: %.2f 秒\n",
                epoch+1, epoch_loss / (float)loader[0]->batch_count, 
                (float)num_matched / (float)sc * 100.0F, elapsed
            );
        }
        
        ema_apply_shadow(ema, model, backup_space);

        if (nn_model_save(model, MODEL_PATH)) {
            printf("模型已保存\n");
        }
    }

    // -----------------------------------PREDICTION----------------------------------- //
    if (nn_model_load(model, MODEL_PATH)) {
        printf("模型已加载\n");

        dataset *image_set = dset_create(global_stack);

        size_t image_count = dset_load_image_folder(
            image_set, "../data/cifar10/test",
            RESIZE(32, 32)|NORMALIZE|STANDARDIZE
        );

        dataloader *image_loader = dloader_create(global_stack, image_set, BATCH_SIZE, 0);

        size_t *preds = (size_t*)obs_alloc(global_stack, sizeof(size_t) * BATCH_SIZE, 1);
    
        while(dloader_iterate(image_loader)) {
            const matrix *output = nn_model_predict(global_stack, model, image_loader->curr_input);
        
            mat_argmax(output, preds);
        
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                size_t image_index = i + image_loader->batch_index * BATCH_SIZE;
            
                if (image_index >= image_count) { // NOSONAR
                    break;
                }

                printf("------------------------------------------\n");

                const matrix *image_view = utils_get_image_view(global_stack, image_set, image_index);
                utils_draw_image(image_view, 32, 32, 3); // Remember to update its content when not using tmux. 💀

                printf(
                    "图片序号: %2zu | 预测结果: [%s]\n",
                    image_index, CIFAR10_LABELS[preds[i]]
                );
            }
        }

        printf("------------------------------------------\n");
    } else {
        printf("🤣👉🤡\n");
    }

    obs_destroy(global_stack);
    obs_destroy_all_markers();

    return 0;
}