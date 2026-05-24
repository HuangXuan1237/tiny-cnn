#pragma once

#include "obs.h"
#include "matrix.h"
#include "nn.h"

#define opt_coslr_step(scheduler, opt) \
    _Generic(opt, \
        adam_optimizer *: _adam_coslr_step, \
        sgd_optimizer *: _sgd_coslr_step \
    )(scheduler, opt)

#define opt_steplr_step(scheduler, opt) \
    _Generic(opt, \
        adam_optimizer *: _adam_steplr_step, \
        sgd_optimizer *: _sgd_steplr_step \
    )(scheduler, opt)

typedef struct adam_optimizer {
    obs *stk;

    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;     // 权重衰减系数
    float t;                // 时间步
    matrix **m;             // 一阶矩缓存数组
    matrix **v;             // 二阶矩缓存数组
} adam_optimizer;

typedef struct sgd_optimizer {
    obs *stk;
    float lr;
    float momentum;
    float weight_decay;
    float dampening;        // 动量阻尼，通常为 0
    bool nesterov;          // 是否使用 Nesterov 动量
    matrix **momentum_buf;  // 动量缓冲区
} sgd_optimizer;

typedef struct {
    obs *stk;
    float decay;
    size_t tensor_count;
    matrix **shadow_values;  // 影子矩阵
    size_t step;
} ema_optimizer;

typedef struct coslr_scheduler {
    float initial_lr;
    float min_lr;
    size_t T_max;
    size_t current_step;
} coslr_scheduler;

typedef struct steplr_scheduler {
    float initial_lr;
    float gamma; // 衰减因子
    size_t step_size;
    size_t current_step;
} steplr_scheduler;

void opt_zero_grad(const nn_model *model);

adam_optimizer *adam_create(obs *stk, const nn_model *model, float lr, float weight_decay);

void adam_step(adam_optimizer *opt, const nn_model *model, size_t batch_size);

void adamw_step(adam_optimizer *opt, const nn_model *model);

sgd_optimizer *sgd_create(obs *stk, const nn_model *model, float lr, float momentum, float weight_decay, bool nesterov);

void sgd_step(const sgd_optimizer *opt, const nn_model *model, size_t batch_size);

ema_optimizer* ema_create(obs *stk, const nn_model *model, float decay);

matrix **ema_create_backup_space(obs *stk, const nn_model *model);

void ema_update(ema_optimizer *ema, const nn_model *model);

void ema_apply_shadow(const ema_optimizer *ema, const nn_model *model, matrix **backup_space);

void ema_restore_weight(const ema_optimizer *ema, const nn_model *model, matrix **backup_space);

coslr_scheduler *coslr_create(obs *stk, float initial_lr, float min_lr, size_t T_max);

void _adam_coslr_step(coslr_scheduler *scheduler, adam_optimizer *opt);
void _sgd_coslr_step(coslr_scheduler *scheduler, sgd_optimizer *opt);

steplr_scheduler *steplr_create(obs *stk, float initial_lr, float gamma, size_t step_size);

void _adam_steplr_step(steplr_scheduler *scheduler, adam_optimizer *opt);
void _sgd_steplr_step(steplr_scheduler *scheduler, sgd_optimizer *opt);