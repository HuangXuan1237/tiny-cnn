#include "optimizer.h"

#include <math.h>
#include <stdlib.h>
#include <sys/types.h>

#define PI 3.14159265358979323846264338327950288419716939937

void opt_zero_grad(const nn_model *model) {
    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        nn_tensor *t = model->backward_graph.tensors[i];
        if (t->grad) {
            mat_fill(t->grad, 0.0F);
        }
    }
}

adam_optimizer *adam_create(arena *ar, const nn_model *model, float lr, float weight_decay) {
    adam_optimizer *opt = (adam_optimizer*)arena_alloc(ar, sizeof(adam_optimizer), 1);

    opt->ar = ar;
    opt->lr = lr;
    opt->beta1 = 0.9F;
    opt->beta2 = 0.999F;
    opt->epsilon = 1E-8F;
    opt->weight_decay = weight_decay;
    opt->t = 0.0F;

    size_t count = model->backward_graph.tensor_count;
    opt->m = (matrix**)arena_alloc(opt->ar, sizeof(matrix*) * count, 1);
    opt->v = (matrix**)arena_alloc(opt->ar, sizeof(matrix*) * count, 1);

    for (size_t i = 0; i < count; i++) {
        const nn_tensor *curr = model->backward_graph.tensors[i];

        if (curr->flags & TENSOR_AS_PARAM) {
            opt->m[i] = mat_create(opt->ar, curr->value->rows, curr->value->cols);
            opt->v[i] = mat_create(opt->ar, curr->value->rows, curr->value->cols);
        }
    }

    return opt;
}

void adam_step(adam_optimizer *opt, const nn_model *model, size_t batch_size) {
    opt->t += 1.0F;
    float beta1_t = powf(opt->beta1, opt->t);
    float beta2_t = powf(opt->beta2, opt->t);
    
    float bias_corr1 = 1.0F - beta1_t;
    float bias_corr2 = 1.0F - beta2_t;

    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        nn_tensor *curr = model->backward_graph.tensors[i];
        if ((curr->flags & TENSOR_AS_PARAM) != TENSOR_AS_PARAM) {
            continue;
        }

        size_t size = curr->value->rows * curr->value->cols;
        float *param = curr->value->data;
        const float *grad = curr->grad->data;
        float *m = opt->m[i]->data;
        float *v = opt->v[i]->data;

        for (size_t k = 0; k < size; k++) {
            float g = grad[k] / (float)batch_size;

            m[k] = opt->beta1 * m[k] + (1.0F - opt->beta1) * g;
            v[k] = opt->beta2 * v[k] + (1.0F - opt->beta2) * g * g;

            float m_hat = m[k] / bias_corr1;
            float v_hat = v[k] / bias_corr2;

            float update = opt->lr * (m_hat / (sqrtf(v_hat) + opt->epsilon) + opt->weight_decay * param[k]);
            param[k] -= update;
        }
    }
}

void adamw_step(adam_optimizer *opt, const nn_model *model) {
    opt->t += 1.0F;
    float lr_t = opt->lr * sqrtf(1.0F - powf(opt->beta2, opt->t)) / (1.0F - powf(opt->beta1, opt->t));

    #pragma omp parallel for
    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        nn_tensor *t = model->backward_graph.tensors[i];

        if (!(t->flags & TENSOR_AS_PARAM) || !t->grad) {
            continue;
        }

        float *w = t->value->data;
        const float *g = t->grad->data;
        float *m = opt->m[i]->data;
        float *v = opt->v[i]->data;
        size_t size = t->value->rows * t->value->cols;

        for (size_t j = 0; j < size; j++) {
            w[j] -= opt->lr * opt->weight_decay * w[j];

            m[j] = opt->beta1 * m[j] + (1.0F - opt->beta1) * g[j];
            v[j] = opt->beta2 * v[j] + (1.0F - opt->beta2) * g[j] * g[j];

            w[j] -= lr_t * m[j] / (sqrtf(v[j]) + opt->epsilon);
        }
    }
}

sgd_optimizer *sgd_create(arena *ar, const nn_model *model, float lr, float momentum, float weight_decay, bool nesterov) {
    sgd_optimizer *opt = (sgd_optimizer*)arena_alloc(ar, sizeof(sgd_optimizer), 1);
    opt->ar = ar;
    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = weight_decay;
    opt->dampening = 0.0F;
    opt->nesterov = nesterov;

    size_t count = model->backward_graph.tensor_count;
    opt->momentum_buf = (matrix**)arena_alloc(ar, sizeof(matrix*) * count, 1);

    for (size_t i = 0; i < count; i++) {
        const nn_tensor *curr = model->backward_graph.tensors[i];

        if (curr->flags & TENSOR_AS_PARAM) {
            opt->momentum_buf[i] = mat_create(ar, curr->value->rows, curr->value->cols);
        }
    }
    
    return opt;
}

void sgd_step(const sgd_optimizer *opt, const nn_model *model, size_t batch_size) {
    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        nn_tensor *curr = model->backward_graph.tensors[i];

        if ((curr->flags & TENSOR_AS_PARAM) != TENSOR_AS_PARAM) continue;

        size_t size = curr->value->rows * curr->value->cols;
        float *param = curr->value->data;
        const float *grad = curr->grad->data;
        float *buf = opt->momentum_buf[i] ? opt->momentum_buf[i]->data : NULL;

        for (size_t k = 0; k < size; k++) {
            float g = grad[k] / (float)batch_size;

            if (opt->weight_decay != 0.0F) {
                g += opt->weight_decay * param[k];
            }

            if (opt->momentum != 0.0F) {
                buf[k] = opt->momentum * buf[k] + (1.0F - opt->dampening) * g; //NOSONAR

                if (opt->nesterov) { //NOSONAR
                    param[k] -= opt->lr * (g + opt->momentum * buf[k]);
                } else {
                    param[k] -= opt->lr * buf[k];
                }
            } else {
                param[k] -= opt->lr * g;
            }
        }
    }
}

ema_optimizer* ema_create(arena *ar, const nn_model *model, float decay) {
    ema_optimizer *ema = (ema_optimizer*)arena_alloc(ar, sizeof(ema_optimizer), 1);
    ema->ar = ar;
    ema->decay = decay;
    ema->tensor_count = model->backward_graph.tensor_count;
    
    ema->shadow_values = (matrix**)arena_alloc(ar, sizeof(matrix*) * ema->tensor_count, 1);
    
    for (size_t i = 0; i < ema->tensor_count; i++) {
        const nn_tensor *t = model->backward_graph.tensors[i];

        if (t->flags & TENSOR_AS_PARAM) {
            ema->shadow_values[i] = mat_create(ar, t->value->rows, t->value->cols);

            memcpy(ema->shadow_values[i]->data, t->value->data, sizeof(float) * t->value->rows * t->value->cols);
        } else {
            ema->shadow_values[i] = NULL;
        }
    }

    ema->step = 0;

    return ema;
}

matrix **ema_create_backup_space(arena *ar, const nn_model *model) {
    size_t count = model->backward_graph.tensor_count;
    
    matrix **backup_space = (matrix**)arena_alloc(ar, sizeof(matrix*) * count, 1);
    
    for (size_t i = 0; i < count; i++) {
        const nn_tensor *t = model->backward_graph.tensors[i];
        
        if (t->flags & TENSOR_AS_PARAM) {
            if (t->value != NULL) {
                backup_space[i] = mat_create(ar, t->value->rows, t->value->cols);
            } else {
                backup_space[i] = NULL;
            }
        } else {
            backup_space[i] = NULL;
        }
    }
    
    return backup_space;
}

void ema_update(ema_optimizer *ema, const nn_model *model) {
    ema->step++;
    
    float current_decay = fminf(ema->decay, (1.0F + (float)ema->step) / (10.0F + (float)ema->step));
    float one_minus_d = 1.0F - current_decay;

    #pragma omp parallel for
    for (size_t i = 0; i < ema->tensor_count; i++) {
        if (ema->shadow_values[i] == NULL) {
            continue;
        }

        const nn_tensor *t = model->backward_graph.tensors[i];
        float *shadow = ema->shadow_values[i]->data;
        const float *current = t->value->data;
        size_t size = t->value->rows * t->value->cols;
        
        for (size_t j = 0; j < size; j++) {
            shadow[j] = current_decay * shadow[j] + one_minus_d * current[j];
        }
    }
}

void ema_apply_shadow(const ema_optimizer *ema, const nn_model *model, matrix **backup_space) {
    for (size_t i = 0; i < ema->tensor_count; i++) {
        if (ema->shadow_values[i] == NULL) {
            continue;
        }
        
        nn_tensor *t = model->backward_graph.tensors[i];
        memcpy(backup_space[i]->data, t->value->data, sizeof(float) * t->value->rows * t->value->cols);
        memcpy(t->value->data, ema->shadow_values[i]->data, sizeof(float) * t->value->rows * t->value->cols);
    }
}

void ema_restore_weight(const ema_optimizer *ema, const nn_model *model, matrix **backup_space) {
    for (size_t i = 0; i < ema->tensor_count; i++) {
        if (ema->shadow_values[i] == NULL) {
            continue;
        }

        nn_tensor *t = model->backward_graph.tensors[i];
        memcpy(t->value->data, backup_space[i]->data, sizeof(float) * t->value->rows * t->value->cols);
    }
}

coslr_scheduler *coslr_create(arena *ar, float initial_lr, float min_lr, size_t T_max) {
    coslr_scheduler *scheduler = (coslr_scheduler*)arena_alloc(ar, sizeof(coslr_scheduler), 1);

    scheduler->initial_lr = initial_lr;
    scheduler->min_lr = min_lr;
    scheduler->T_max = T_max;
    scheduler->current_step = 0;

    return scheduler;
}

void _adam_coslr_step(coslr_scheduler *scheduler, adam_optimizer *opt) {
    float lr;
    
    if (scheduler->current_step >= scheduler->T_max) {
        lr = scheduler->min_lr;
    } else {
        float cos_arg = (float)(PI * (float)scheduler->current_step / (float)scheduler->T_max);
        lr = scheduler->min_lr + 0.5F * (scheduler->initial_lr - scheduler->min_lr) * (1.0F + cosf(cos_arg));
    }

    opt->lr = lr;

    scheduler->current_step++;
}

void _sgd_coslr_step(coslr_scheduler *scheduler, sgd_optimizer *opt) {
    float lr;
    
    if (scheduler->current_step >= scheduler->T_max) {
        lr = scheduler->min_lr;
    } else {
        float cos_arg = (float)(PI * (float)scheduler->current_step / (float)scheduler->T_max);
        lr = scheduler->min_lr + 0.5F * (scheduler->initial_lr - scheduler->min_lr) * (1.0F + cosf(cos_arg));
    }

    opt->lr = lr;

    scheduler->current_step++;
}

steplr_scheduler *steplr_create(arena *ar, float initial_lr, float gamma, size_t step_size) {
    steplr_scheduler *scheduler = (steplr_scheduler*)arena_alloc(ar, sizeof(steplr_scheduler), 1);
    scheduler->initial_lr = initial_lr;
    scheduler->gamma = gamma;
    scheduler->step_size = step_size;
    scheduler->current_step = 0;

    return scheduler;
}

void _adam_steplr_step(steplr_scheduler *scheduler, adam_optimizer *opt) {
    scheduler->current_step++;
    if (scheduler->current_step % scheduler->step_size == 0) {
        opt->lr *= scheduler->gamma;
    }
}

void _sgd_steplr_step(steplr_scheduler *scheduler, sgd_optimizer *opt) {
    scheduler->current_step++;
    if (scheduler->current_step % scheduler->step_size == 0) {
        opt->lr *= scheduler->gamma;
    }
}