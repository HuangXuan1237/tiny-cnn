#include <time.h>
#include "stack.h"
#include "data_utils.h"
#include "matrix.h"
#include "nn.h"
#include "optimizer.h"
#include "utils.h"

#define NUM_SELECTED 5
#define BATCH_SIZE 64
#define EPOCHS 50
#define LEARNING_RATE 5E-4F

#define PRE_TRAINED 0
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

void cifar10_custom_resnet(stack *stk, nn_model *model) {
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
    get_system_entropy(seeds, sizeof(seeds));

    pcg32_set_seed(seeds[0], seeds[1]);

    openblas_set_num_threads(1); // Hand over all the threading to OpenMP.

    stack *global_stack = stack_create();

    dataset *cifar10_dataset[2];
    cifar10_dataset[0] = dset_create(global_stack);
    cifar10_dataset[1] = dset_create(global_stack);

    dset_load(
        cifar10_dataset[0], cifar10_dataset[1], CIFAR10,
        NORMALIZE|STANDARDIZE|RANDOM_HFLIP(0.5)|RANDOM_CROP(4)
    );
     
    dataloader *loader[2];
    loader[0] = dloader_create(global_stack, cifar10_dataset[0], BATCH_SIZE, 1);
    loader[1] = dloader_create(global_stack, cifar10_dataset[1], BATCH_SIZE, 0);

    dloader_apply_label_smoothing(loader[0], 0.1F, 10);

    nn_model *model = nn_model_create(global_stack);
    
    cifar10_custom_resnet(global_stack, model);
    nn_model_compile(global_stack, model);

    if (!PRE_TRAINED) {
        adam_optimizer *opt = adam_create(global_stack, model, LEARNING_RATE, 1E-4F);

        ema_optimizer *ema = ema_create(global_stack, model, 0.999F); 

        matrix **backup_space = ema_create_backup_space(global_stack, model);

        coslr_scheduler *scheduler = coslr_create(global_stack, LEARNING_RATE, 1E-5F, EPOCHS);    

        struct timespec start, end; // NOSONAR
        
        size_t *preds = stack_alloc(global_stack, sizeof(size_t) * BATCH_SIZE, 1);
        size_t *targets = stack_alloc(global_stack, sizeof(size_t) * BATCH_SIZE, 1);

        for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
            clock_gettime(CLOCK_MONOTONIC, &start);

            // -----------------------------------TRAINING----------------------------------- //
            float epoch_loss = 0;

            while (dloader_iterate(loader[0])) {
                opt_zero_grad(model);

                float batch_loss = nn_model_criterion(
                    model, loader[0]->curr_input, loader[0]->curr_target
                ) / (float)BATCH_SIZE;

                adam_step(opt, model, BATCH_SIZE);
            
                ema_update(ema, model);

                epoch_loss += batch_loss;

                if (loader[0]->curr_batch % 10 == 0) { // NOSONAR
                    printf(
                        "EPOCH %2zd/%2d | BATCH %3zd/%3zd | LOSS: %.4F\r",
                        epoch + 1, EPOCHS, loader[0]->curr_batch, loader[0]->batch_count, batch_loss
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
                "\rEPOCH %2zd | AVG LOSS: %.4F | 验证准确率: %.2F%% | 耗时: %.2F 秒\n",
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
    }

    size_t image_count, batch_count; // NOSONAR
    matrix **imageset = load_imageset(
        global_stack, "../data/cifar10/test", 
        32, 32, 3, BATCH_SIZE,
        &image_count, &batch_count
    );

    size_t *preds = (size_t*)stack_alloc(global_stack, sizeof(size_t) * image_count, 1);
    for (size_t batch = 0; batch < batch_count; batch++) {
        matrix *images = imageset[batch];

        matrix *processed_images = mat_create(global_stack, images->rows, images->cols);
        standardize_image(processed_images, images, 32, 32 ,3);

        const matrix *output = nn_model_predict(global_stack, model, processed_images);
        mat_argmax(output, preds + batch * BATCH_SIZE);

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            size_t image_index = i + batch * BATCH_SIZE;
            if (image_index == image_count) {
                break;
            }

            printf("------------------------------------------\n");

            matrix image_view = {
                .rows = 1,
                .cols = 32 * 32 * 3,
                .data = images->data + i * 32 * 32 * 3
            };

            // Remember to update its content when using tmux in your terminal. XD
            draw_image(&image_view, 32, 32, 3);

            printf(
                "图片序号: %2zu | 预测结果: [%s]\n",
                image_index, CIFAR10_LABELS[preds[image_index]]
            );
        }
    }
    printf("------------------------------------------\n");

    stack_destroy(global_stack);

    return 0;
}