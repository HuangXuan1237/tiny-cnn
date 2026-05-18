#include "nn.h"

#define NN_GET_INPUT_COUNT(op) ( \
    (op) < _TENSOR_OP_UNARY  ? 0 : \
    (op) < _TENSOR_OP_BINARY ? 1 : \
    (op) != TENSOR_OP_BATCHNORM2D ? 2 : 3 \
)

nn_tensor* nn_tensor_create(
    stack* stk, nn_model* model,
    size_t c, size_t h, size_t w,
    size_t batch_size, uint32_t flags
) {
    nn_tensor* tensor = (nn_tensor*)stack_alloc(stk, sizeof(nn_tensor), 1);

    tensor->c = c; 
    tensor->h = h; 
    tensor->w = w;
    
    tensor->index = model->tensor_count++;
    tensor->flags = flags;
    tensor->op = TENSOR_OP_NONE;

    size_t rows = (flags & TENSOR_AS_PARAM) ? 1 : batch_size;
    tensor->value = mat_create(stk, rows, c * h * w);

    if (flags & TENSOR_REQUIRES_GRAD) {
        tensor->grad = mat_create(stk, rows, c * h * w);
    }

    if (flags & TENSOR_AS_INPUT) {
        model->input = tensor;
    }

    if (flags & TENSOR_AS_OUTPUT) {
        model->output = tensor;
    }

    if (flags & TENSOR_AS_TARGET) {
        model->target = tensor;
    }

    if (flags & TENSOR_AS_LOSS) {
        model->loss = tensor;
    }

    return tensor;
}

static nn_tensor* _nn_tensor_unary_impl( //NOSONAR
    stack* stk, nn_model* model,
    nn_tensor* input, size_t c, size_t h, size_t w,
    uint32_t flags, nn_tensor_ops op
) {
    if (input->flags & TENSOR_REQUIRES_GRAD) {
        flags |= TENSOR_REQUIRES_GRAD;
    }

    size_t batch_size = input->value->rows; 
    nn_tensor* tensor = nn_tensor_create(stk, model, c, h, w, batch_size, flags);

    tensor->op = op;
    tensor->inputs[0] = input;

    return tensor;
}

nn_tensor* nn_func_dropout(
    stack* stk, nn_model* model,
    nn_tensor* input, float p, uint32_t flags
) {
    if (p < 0.0f || p >= 1.0f) {
        return NULL; //无效概率
    }

    if (input->flags & TENSOR_REQUIRES_GRAD) {
        flags |= TENSOR_REQUIRES_GRAD;
    }

    size_t batch_size = input->value->rows;
    size_t elements = input->c * input->h * input->w;

    nn_tensor* tensor = nn_tensor_create(stk, model, input->c, input->h, input->w, batch_size, flags);
    tensor->op = TENSOR_OP_DROPOUT;
    tensor->inputs[0] = input;

    tensor->eps = p;

    tensor->aux = mat_create(stk, batch_size, elements);

    return tensor;
}

nn_tensor* nn_func_maxpool2d(
    stack* stk, nn_model* model,
    nn_tensor* input, size_t k_size, size_t stride,
    uint32_t flags
) {
    size_t out_c = input->c;
    size_t out_h = (input->h - k_size) / stride + 1;
    size_t out_w = (input->w - k_size) / stride + 1;

    nn_tensor* tensor = _nn_tensor_unary_impl(
        stk, model, input,
        out_c, out_h, out_w,
        flags, TENSOR_OP_MAXPOOL2D
    );

    tensor->k_size = k_size;
    tensor->stride = stride;
    tensor->padding = 0;

    return tensor;
}

nn_tensor* nn_func_gapool2d(
    stack* stk, nn_model* model,
    nn_tensor* input, uint32_t flags
) {
    size_t c = input->c;

    return _nn_tensor_unary_impl(
        stk, model, input,
        c, 1, 1, flags, TENSOR_OP_GAPOOL2D
    );
}

nn_tensor* nn_func_relu(
    stack* stk, nn_model* model,
    nn_tensor* input, uint32_t flags
) {
    return _nn_tensor_unary_impl(
        stk, model, input,
        input->c, input->h, input->w,
        flags, TENSOR_OP_RELU
    );
}

nn_tensor* nn_func_softmax(
    stack* stk, nn_model* model,
    nn_tensor* input, uint32_t flags
) {
    return _nn_tensor_unary_impl(
        stk, model, input, 
        input->c, input->h, input->w,
        flags, TENSOR_OP_SOFTMAX
    );
}

static nn_tensor* _nn_tensor_binary_impl( //NOSONAR
    stack* stk, nn_model* model,
    nn_tensor* x, nn_tensor* y,
    size_t c, size_t h, size_t w,
    uint32_t flags, nn_tensor_ops op
) {
    if ((x->flags & TENSOR_REQUIRES_GRAD) ||
        (y->flags & TENSOR_REQUIRES_GRAD)) {
        flags |= TENSOR_REQUIRES_GRAD;
    }

    size_t batch_size = (x->flags & TENSOR_AS_PARAM) ? y->value->rows : x->value->rows;
    nn_tensor* tensor = nn_tensor_create(stk, model, c, h, w, batch_size, flags);

    tensor->op = op;
    tensor->inputs[0] = x;
    tensor->inputs[1] = y;

    return tensor;
}

nn_tensor* nn_func_add(
    stack* stk, nn_model* model,
    nn_tensor* x, nn_tensor* y, uint32_t flags
) {
    if (x->c != y->c) {
        return NULL;
    }
    
    size_t out_h;
    size_t out_w;

    if (x->h == y->h && x->w == y->w ||y->h == 1 && y->w == 1) {
        out_h = x->h;
        out_w = x->w;
    } else if (x->h == 1 && x->w == 1) {
        out_h = y->h;
        out_w = y->w;
    } else {
        return NULL;
    }

    return _nn_tensor_binary_impl(
        stk, model, x, y,
        x->c, out_h, out_w,
        flags, TENSOR_OP_ADD
    );
}

nn_tensor* nn_func_conv2d( //NOSONAR
    stack* stk, nn_model* model,
    nn_tensor* input, nn_tensor* kernel, 
    size_t k_size, size_t stride, size_t padding,
    uint32_t flags
) {
    if (input == NULL || kernel == NULL) {
        return NULL;
    }

    size_t in_c = input->c;
    size_t kte = kernel->value->rows * kernel->value->cols;
    size_t sfe = in_c * k_size * k_size;

    size_t out_c = kte / sfe; //NOSONAR

    size_t out_h = (input->h + 2 * padding - k_size) / stride + 1;
    size_t out_w = (input->w + 2 * padding - k_size) / stride + 1;

    nn_tensor* tensor = _nn_tensor_binary_impl(
        stk, model, input, kernel, 
        out_c, out_h, out_w, 
        flags, TENSOR_OP_CONV2D
    );

    if (tensor) {
        tensor->stride = stride;
        tensor->padding = padding;
        tensor->k_size = k_size;
    }

    return tensor;
}

nn_tensor* nn_func_cross_entropy(
    stack* stk, nn_model* model,
    nn_tensor* p, nn_tensor* q, uint32_t flags
) {
    size_t p_size = p->c * p->h * p->w;
    size_t q_size = q->c * q->h * q->w;

    if (p_size != q_size) {
        return NULL;
    }

    if ((p->flags & TENSOR_REQUIRES_GRAD) ||
        (q->flags & TENSOR_REQUIRES_GRAD)) {
        flags |= TENSOR_REQUIRES_GRAD;
    }
    
    size_t batch_size = p->value->rows;
    nn_tensor* tensor = nn_tensor_create(stk, model, 1, 1, 1, batch_size, flags);
    
    tensor->op = TENSOR_OP_CROSS_ENTROPY;
    tensor->inputs[0] = p;
    tensor->inputs[1] = q;

    return tensor;
}

nn_tensor* nn_func_matmul(
    stack* stk, nn_model* model,
    nn_tensor* x, nn_tensor* y, uint32_t flags
) {    
    size_t x_size = x->c * x->h * x->w;

    if (x_size != y->h) {
        return NULL;
    }

    if ((x->flags & TENSOR_REQUIRES_GRAD) ||
        (y->flags & TENSOR_REQUIRES_GRAD)) {
        flags |= TENSOR_REQUIRES_GRAD;
    }

    size_t batch_size = x->value->rows;
    nn_tensor* tensor = nn_tensor_create(
        stk, model, 1, 1, y->w, batch_size, flags
    );

    tensor->op = TENSOR_OP_MATMUL;
    tensor->inputs[0] = x;
    tensor->inputs[1] = y;

    return tensor;
}

nn_tensor* nn_func_batchnorm2d( //NOSONAR
    stack* stk, nn_model* model,
    nn_tensor* input, nn_tensor* gamma, nn_tensor* beta,
    float eps, float momentum,
    uint32_t flags
) {
    if (gamma->c != input->c || gamma->h != 1 || gamma->w != 1 ||
        beta->c != input->c || beta->h != 1 || beta->w != 1) {
        return NULL;
    }

    size_t batch_size = input->value->rows;
    size_t c = input->c;
    size_t h = input->h;
    size_t w = input->w;
    size_t spatial = h * w;

    nn_tensor* tensor = (nn_tensor*)stack_alloc(stk, sizeof(nn_tensor), 1);

    tensor->c = c;
    tensor->h = h;
    tensor->w = w;
    tensor->index = model->tensor_count++;
    
    uint32_t f = flags;
    if ((input->flags & TENSOR_REQUIRES_GRAD) ||
        (gamma->flags & TENSOR_REQUIRES_GRAD) ||
        (beta->flags & TENSOR_REQUIRES_GRAD)
    ) {
        f |= TENSOR_REQUIRES_GRAD;
    }
    tensor->flags = f;
    
    tensor->op = TENSOR_OP_BATCHNORM2D;
    tensor->eps = eps;
    tensor->momentum = momentum;
    tensor->inputs[0] = input;
    tensor->inputs[1] = gamma;
    tensor->inputs[2] = beta;

    tensor->value = mat_create(stk, batch_size, c * spatial);
    if (f & TENSOR_REQUIRES_GRAD) {
        tensor->grad = mat_create(stk, batch_size, c * spatial);
    }

    tensor->running_mean = mat_create(stk, 1, c);
    mat_fill(tensor->running_mean, 0.0f);
    tensor->running_var = mat_create(stk, 1, c);
    mat_fill(tensor->running_var, 1.0f);

    tensor->aux = mat_create(stk, batch_size, c * spatial);
    tensor->aux2 = mat_create(stk, 1, c);

    return tensor;
}

nn_tensor *nn_layer_input(
    stack* stk, nn_model* model,
    size_t c, size_t h, size_t w,
    size_t batch_size
) {
    return nn_tensor_create(stk, model, c, h, w, batch_size, TENSOR_AS_INPUT);
}

nn_tensor *nn_layer_target(
    stack* stk, nn_model* model,
    size_t c, size_t h, size_t w,
    size_t batch_size
) {
    return nn_tensor_create(stk, model, c, h, w, batch_size, TENSOR_AS_TARGET);
}

nn_tensor *nn_layer_batchnorm2d(
    stack *stk, nn_model *model, nn_tensor *input
) {
    size_t oc = input->c; 

    nn_tensor *gamma = nn_tensor_create(stk, model, oc, 1, 1, 1, TENSOR_REQUIRES_GRAD|TENSOR_AS_PARAM);
    mat_fill(gamma->value, 1.0f);

    nn_tensor *beta = nn_tensor_create(stk, model, oc, 1, 1, 1, TENSOR_REQUIRES_GRAD|TENSOR_AS_PARAM);
    mat_fill(beta->value, 0.0f);

    return nn_func_batchnorm2d(stk, model, input, gamma, beta, 1e-5f, 0.9f, 0);
}

nn_tensor *nn_layer_conv2d(
    stack *stk, nn_model *model,
    nn_tensor *input, 
    size_t out_channels, size_t k_size,
    size_t stride, size_t padding
) {
    size_t fi = input->c * k_size * k_size;
    nn_tensor *w = nn_tensor_create(
        stk, model, out_channels, input->c, k_size*k_size, 1,
        TENSOR_REQUIRES_GRAD|TENSOR_AS_PARAM
    );
    nn_tensor *b = nn_tensor_create(
        stk, model, out_channels, 1, 1, 1,
        TENSOR_REQUIRES_GRAD|TENSOR_AS_PARAM
    );

    float std = sqrtf(2.0f / (float)fi);
    size_t total = w->value->rows * w->value->cols;

    for (size_t idx = 0; idx < total; idx++) {
        w->value->data[idx] = pcg32_gaussian() * std;
    }

    mat_fill(b->value, 0.0f);

    nn_tensor *conv = nn_func_conv2d(
        stk, model, input, w, k_size, stride, padding, 0
    );

    return nn_func_add(stk, model, conv, b, 0);
}

nn_tensor *nn_layer_cross_entropy(
    stack *stk, nn_model *model, nn_tensor *pred, nn_tensor *target
) {
    return nn_func_cross_entropy(stk, model, pred, target, TENSOR_AS_LOSS);
}

nn_tensor *nn_layer_dropout(
    stack *stk, nn_model *model, nn_tensor *input, float prob
) {
    return nn_func_dropout(stk, model, input, prob, 0);
}

nn_tensor *nn_layer_gapool2d(
    stack *stk, nn_model *model, nn_tensor *input
) {
    return nn_func_gapool2d(stk, model, input, 0);
}

nn_tensor *nn_layer_linear(
    stack *stk, nn_model *model, nn_tensor *input, size_t out_features
) {
    size_t fs = input->c * input->h * input->w;

    nn_tensor *w = nn_tensor_create(
        stk, model, 1, fs, out_features, 1,
        TENSOR_REQUIRES_GRAD|TENSOR_AS_PARAM
    );
    nn_tensor *b = nn_tensor_create(
        stk, model, 1, 1, out_features, 1,
        TENSOR_REQUIRES_GRAD|TENSOR_AS_PARAM
    );

    float std = sqrtf(2.0f / (float)fs);
    size_t total = w->value->rows * w->value->cols;

    for (size_t idx = 0; idx < total; idx++) {
        w->value->data[idx] = pcg32_gaussian() * std;
    }

    mat_fill(b->value, 0.0f); 

    nn_tensor *z = nn_func_matmul(stk, model, input, w, 0);

    return nn_func_add(stk, model, z, b, 0);
}

nn_tensor *nn_layer_maxpool2d(
    stack *stk, nn_model * model, nn_tensor *input,
    size_t k_size, size_t stride
) {
    return nn_func_maxpool2d(stk, model, input, k_size, stride, 0);
}

nn_tensor *nn_layer_relu(stack *stk, nn_model *model, nn_tensor *input) {
    return nn_func_relu(stk, model, input, 0);
}

nn_tensor* nn_layer_residual_block(
    stack* stk, nn_model* model,
    nn_tensor* input,
    size_t in_planes, size_t out_planes,
    size_t stride
) {
    nn_tensor* w1 = nn_tensor_create(stk, model, out_planes, in_planes, 9, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
    float std1 = sqrtf(2.0f / ((float)in_planes * 9));
    size_t total1 = w1->value->rows * w1->value->cols;
    for (size_t idx = 0; idx < total1; idx++) {
        w1->value->data[idx] = pcg32_gaussian() * std1;
    }

    nn_tensor* conv1 = nn_func_conv2d(stk, model, input, w1, 3, stride, 1, 0);

    nn_tensor* gamma1 = nn_tensor_create(stk, model, out_planes, 1, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
    mat_fill(gamma1->value, 1.0f);
    nn_tensor* beta1 = nn_tensor_create(stk, model, out_planes, 1, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
    mat_fill(beta1->value, 0.0f);
    nn_tensor* bn1 = nn_func_batchnorm2d(stk, model, conv1, gamma1, beta1, 1e-5f, 0.9f, 0);
    nn_tensor* relu1 = nn_func_relu(stk, model, bn1, 0);

    nn_tensor* w2 = nn_tensor_create(stk, model, out_planes, out_planes, 9, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
    float std2 = sqrtf(2.0f / ((float)out_planes * 9));
    size_t total2 = w2->value->rows * w2->value->cols;
    for (size_t idx = 0; idx < total2; idx++) {
        w2->value->data[idx] = pcg32_gaussian() * std2;
    }

    nn_tensor* conv2 = nn_func_conv2d(stk, model, relu1, w2, 3, 1, 1, 0);

    nn_tensor* gamma2 = nn_tensor_create(stk, model, out_planes, 1, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
    mat_fill(gamma2->value, 1.0f);
    nn_tensor* beta2 = nn_tensor_create(stk, model, out_planes, 1, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
    mat_fill(beta2->value, 0.0f);
    nn_tensor* bn2 = nn_func_batchnorm2d(stk, model, conv2, gamma2, beta2, 1e-5f, 0.9f, 0);

    nn_tensor* shortcut;
    if (stride != 1 || in_planes != out_planes) {
        nn_tensor* w_proj = nn_tensor_create(stk, model, out_planes, in_planes, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
        float std_proj = sqrtf(2.0f / (float)in_planes);
        size_t total_proj = w_proj->value->rows * w_proj->value->cols;
        for (size_t idx = 0; idx < total_proj; idx++) {
            w_proj->value->data[idx] = pcg32_gaussian() * std_proj;
        }

        nn_tensor* conv_proj = nn_func_conv2d(stk, model, input, w_proj, 1, stride, 0, 0);

        nn_tensor* gamma_proj = nn_tensor_create(stk, model, out_planes, 1, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
        mat_fill(gamma_proj->value, 1.0f);
        nn_tensor* beta_proj = nn_tensor_create(stk, model, out_planes, 1, 1, 1, TENSOR_REQUIRES_GRAD | TENSOR_AS_PARAM);
        mat_fill(beta_proj->value, 0.0f);
        shortcut = nn_func_batchnorm2d(stk, model, conv_proj, gamma_proj, beta_proj, 1e-5f, 0.9f, 0);
    } else {
        shortcut = input;
    }

    nn_tensor* add = nn_func_add(stk, model, bn2, shortcut, 0);
    nn_tensor* out = nn_func_relu(stk, model, add, 0);

    return out;
}

nn_tensor *nn_layer_softmax(stack *stk, nn_model *model, nn_tensor *input) {
    return nn_func_softmax(stk, model, input, TENSOR_AS_OUTPUT);
}

static void _forward(const _nn_graph* graph, bool is_training) {
    for (size_t i = 0; i < graph->tensor_count; i ++) {
        nn_tensor* curr = graph->tensors[i];

        nn_tensor* x = curr->inputs[0];
        nn_tensor* y = curr->inputs[1];

        switch (curr->op) {
            case TENSOR_OP_DROPOUT: {
                const nn_tensor* input = curr->inputs[0];
                float p = curr->eps;
                float keep_prob = 1.0f - p;
                size_t total_elements = curr->value->rows * curr->value->cols;
            
                if (is_training) {
                    for (size_t i = 0; i < total_elements; i++) { //NOSONAR
                        float r = pcg32_randomf();
                        if (r < keep_prob) {
                            curr->value->data[i] = input->value->data[i] / keep_prob;
                            curr->aux->data[i] = 1.0f / keep_prob;
                        } else {
                            curr->value->data[i] = 0.0f;
                            curr->aux->data[i] = 0.0f;
                        }
                    }
                } else {
                    memcpy(curr->value->data, input->value->data, total_elements * sizeof(float));
                }
            } break;

            case TENSOR_OP_MAXPOOL2D: {
                mat_maxpool2d(
                    curr->value, x->value,
                    x->c, x->h, x->w,
                    curr->k_size, curr->stride
                );
            } break;

            case TENSOR_OP_GAPOOL2D: {
                mat_gavgpool2d(curr->value, x->value, x->c, x->h, x->w);
            } break;

            case TENSOR_OP_RELU: {
                mat_relu(curr->value, x->value);
            } break;

            case TENSOR_OP_SOFTMAX: {
                mat_softmax(curr->value, x->value);
            } break;

            case TENSOR_OP_ADD: {
                mat_add(curr->value, x->value, y->value);
            } break;

            case TENSOR_OP_CONV2D: {
                mat_conv2d(
                    curr->value, x->value,
                    y->value, x->c, x->h, x->w,
                    curr->k_size, curr->stride, curr->padding
                );
            } break;

            case TENSOR_OP_CROSS_ENTROPY: {
                mat_cross_entropy(curr->value, x->value, y->value);
            } break;

            case TENSOR_OP_MATMUL: {
                size_t in_size = x->c * x->h * x->w;
                size_t out_size = (y->c * y->h * y->w) / in_size;

                y->value->rows = in_size;
                y->value->cols = out_size;

                mat_mul(curr->value, x->value, y->value, 0.0f, 0, 0);

                y->value->rows = 1;
                y->value->cols = in_size * out_size;
            } break;

            case TENSOR_OP_BATCHNORM2D: {
                const nn_tensor* gamma = curr->inputs[1];
                const nn_tensor* beta = curr->inputs[2];
                mat_batchnorm2d(
                    curr->value, x->value, gamma->value, beta->value,
                    curr->running_mean, curr->running_var,
                    curr->eps, curr->momentum, is_training,
                    curr->aux, curr->aux2,
                    curr->c, curr->h * curr->w
                );
            } break;

            default: break;
        }
    }
}

static void _backward(const _nn_graph* graph) { //NOSONAR
    mat_fill(graph->tensors[graph->tensor_count-1]->grad, 1.0f);

    for (size_t i = graph->tensor_count; i-- > 0;) {
        nn_tensor* curr = graph->tensors[i];

        size_t input_count = NN_GET_INPUT_COUNT(curr->op);

        if (input_count == 0) {
            continue;
        }

        nn_tensor* x = curr->inputs[0];
        nn_tensor* y = curr->inputs[1];

        if (
            (x->flags & TENSOR_REQUIRES_GRAD) != TENSOR_REQUIRES_GRAD &&
            input_count == 1
        ) {
            continue;
        }

        if (
            (x->flags & TENSOR_REQUIRES_GRAD) != TENSOR_REQUIRES_GRAD &&
            (y->flags & TENSOR_REQUIRES_GRAD) != TENSOR_REQUIRES_GRAD &&
            input_count == 2
        ) {
            continue;
        }

        switch (curr->op) {
            case TENSOR_OP_DROPOUT: {
                nn_tensor* input = curr->inputs[0];
                if (input->flags & TENSOR_REQUIRES_GRAD) {
                    size_t total_elements = curr->grad->rows * curr->grad->cols;
                    const float* grad_out = curr->grad->data;
                    const float* mask = curr->aux->data;
                    float* grad_in = input->grad->data;
            
                    for (size_t i = 0; i < total_elements; i++) { //NOSONAR
                        grad_in[i] += grad_out[i] * mask[i];
                    }
                }
            } break;

            case TENSOR_OP_MAXPOOL2D: {
                if (x->flags & TENSOR_REQUIRES_GRAD) {
                    mat_grad_maxpool2d(
                        x->grad, x->value, curr->grad,
                        x->c, x->h, x->w,
                        curr->k_size, curr->stride
                    );
                }
            } break;

            case TENSOR_OP_GAPOOL2D: {
                if (x->flags & TENSOR_REQUIRES_GRAD) {
                    mat_grad_gavgpool2d(x->grad, curr->grad, x->c, x->h, x->w);
                }
            } break;

            case TENSOR_OP_RELU: {
                mat_grad_relu(x->grad, x->value, curr->grad);                
            } break;

            case TENSOR_OP_SOFTMAX: {
                mat_grad_softmax(x->grad, curr->value, curr->grad);
            } break;
            
            case TENSOR_OP_ADD: {
                if (x->flags & TENSOR_REQUIRES_GRAD) {
                    if (x->value->rows == 1) { //NOSONAR
                        size_t n = x->value->cols;
                        float* xg = x->grad->data;
                        const float* cg = curr->grad->data;
                        size_t batch_size = curr->grad->rows;
            
                        if (curr->grad->cols == n) {
                            for (size_t j = 0; j < n; j++) {
                                float sum = 0.0f;
                                for (size_t b = 0; b < batch_size; b++) {
                                    sum += cg[b * n + j];
                                }
                                xg[j] += sum;
                            }
                        } else if (curr->grad->cols % n == 0) {
                            size_t repeat = curr->grad->cols / n;
                            for (size_t j = 0; j < n; j++) {
                                float sum = 0.0f;
                                for (size_t b = 0; b < batch_size; b++) {
                                    for (size_t r = 0; r < repeat; r++) {
                                        sum += cg[b * curr->grad->cols + j * repeat + r];
                                    }
                                }
                                xg[j] += sum;
                            }
                        }
                    } else {
                        mat_add(x->grad, x->grad, curr->grad);
                    }
                }
            
                if (y->flags & TENSOR_REQUIRES_GRAD) {
                    if (y->value->rows == 1) { //NOSONAR
                        size_t n = y->value->cols;
                        float* yg = y->grad->data;
                        const float* cg = curr->grad->data;
                        size_t batch_size = curr->grad->rows;
            
                        if (curr->grad->cols == n) {
                            for (size_t j = 0; j < n; j++) {
                                float sum = 0.0f;
                                for (size_t b = 0; b < batch_size; b++) {
                                    sum += cg[b * n + j];
                                }
                                yg[j] += sum;
                            }
                        } else if (curr->grad->cols % n == 0) {
                            size_t repeat = curr->grad->cols / n;
                            for (size_t j = 0; j < n; j++) {
                                float sum = 0.0f;
                                for (size_t b = 0; b < batch_size; b++) {
                                    for (size_t r = 0; r < repeat; r++) {
                                        sum += cg[b * curr->grad->cols + j * repeat + r];
                                    }
                                }
                                yg[j] += sum;
                            }
                        }
                    } else {
                        mat_add(y->grad, y->grad, curr->grad);
                    }
                }
            } break;

            case TENSOR_OP_CONV2D: {
                if (x->flags & TENSOR_REQUIRES_GRAD) {
                    mat_grad_conv2d(
                        x->grad, NULL,
                        x->value, y->value, curr->grad,
                        x->c, x->h, x->w,
                        curr->k_size, curr->stride, curr->padding
                    );
                }

                if (y->flags & TENSOR_REQUIRES_GRAD) {
                    mat_grad_conv2d(
                        NULL, y->grad,
                        x->value, y->value, curr->grad,
                        x->c, x->h, x->w,
                        curr->k_size, curr->stride, curr->padding
                    );
                }
            } break;

            case TENSOR_OP_CROSS_ENTROPY: {
                mat_grad_cross_entropy(
                    x->grad, y->grad, x->value, y->value, curr->grad
                );
            } break;

            case TENSOR_OP_MATMUL: {
                size_t in_size = x->c * x->h * x->w;
                size_t out_size = y->c * y->h * y->w / in_size;

                if (x->flags & TENSOR_REQUIRES_GRAD) {
                    y->value->rows = in_size;
                    y->value->cols = out_size;

                    mat_mul(
                        x->grad, curr->grad, y->value, 1.0f, 0, 1
                    ); 

                    y->value->rows = 1;
                    y->value->cols = in_size * out_size;
                }

                if (y->flags & TENSOR_REQUIRES_GRAD) {
                    y->grad->rows = in_size;
                    y->grad->cols = out_size;
                    
                    mat_mul(
                        y->grad, x->value, curr->grad, 1.0f, 1, 0
                    );
                    
                    y->grad->rows = 1;
                    y->grad->cols = in_size * out_size;
                }                
            } break;

            case TENSOR_OP_BATCHNORM2D: {
                nn_tensor* input = curr->inputs[0];
                nn_tensor* gamma = curr->inputs[1];
                nn_tensor* beta = curr->inputs[2];

                mat_grad_batchnorm2d(
                    (input->flags & TENSOR_REQUIRES_GRAD) ? input->grad : NULL,
                    (gamma->flags & TENSOR_REQUIRES_GRAD) ? gamma->grad : NULL,
                    (beta->flags & TENSOR_REQUIRES_GRAD) ? beta->grad : NULL,
                    input->value, gamma->value, curr->aux, curr->aux2,
                    curr->grad, curr->c, curr->h * curr->w
                );
            } break;

            default: break;
        }
    }
}

nn_model* nn_model_create(stack* stk) {
    return (nn_model*)stack_alloc(stk, sizeof(nn_model), 1);
}

static void __nn_graph_dfs(nn_tensor* node, bool* visited, nn_tensor** graph, size_t* size) {
    if (visited[node->index]) return;
    
    size_t input_count = NN_GET_INPUT_COUNT(node->op);
    for (size_t i = 0; i < input_count; i++) {
        if (node->inputs[i]) {
            __nn_graph_dfs(node->inputs[i], visited, graph, size);
        }
    }
    
    visited[node->index] = 1;
    
    graph[(*size)++] = node;
}

static _nn_graph __nn_graph_create(stack* stk, const nn_model* model, nn_tensor* node) {
    stack_marker scratch = stack_get_marker(&stk, 1);

    bool* visited = (bool*)stack_alloc(scratch.stk, sizeof(bool) * model->tensor_count, 1);

    nn_tensor** graph_tensors = (nn_tensor**)stack_alloc(scratch.stk, sizeof(nn_tensor*) * model->tensor_count, 1);
    size_t graph_size = 0;

    __nn_graph_dfs(node, visited, graph_tensors, &graph_size);

    _nn_graph graph = {
        .tensor_count = graph_size,
        .tensors = stack_alloc(stk, sizeof(nn_tensor*) * graph_size, 0)
    };

    memcpy(graph.tensors, graph_tensors, sizeof(nn_tensor*) * graph_size);

    stack_drop_marker(scratch);
    return graph;
}

void nn_model_compile(stack* stk, nn_model* model) {
    if (model->output != NULL) {
        model->forward_graph = __nn_graph_create(
            stk, model, model->output
        );
    }

    if (model->loss != NULL) {
        model->backward_graph = __nn_graph_create(
            stk, model, model->loss
        );
    }
}

float nn_model_criterion(nn_model* model, matrix* input, matrix* target) {
    float* curr_input = model->input->value->data;
    float* curr_target = model->target->value->data;

    model->input->value->data = input->data;
    model->target->value->data = target->data;

    _forward(&model->backward_graph, 1);
    _backward(&model->backward_graph);

    model->input->value->data = curr_input;
    model->target->value->data = curr_target;

    return mat_sum(model->loss->value);
}

matrix* nn_model_predict(stack *stk, nn_model* model, matrix* input) {
    size_t batch_rows = model->input->value->rows;
    size_t input_rows = input->rows;
    size_t input_cols = input->cols;

    float* checkpoint = model->input->value->data;

    matrix* output = NULL;

    if (input_rows == batch_rows) {
        model->input->value->data = input->data;
        _forward(&model->forward_graph, 0);

        output = model->output->value;
    }
    
    else if (input_rows < batch_rows) {
        stack_marker scratch = stack_get_marker(&stk, 1);
        
        matrix* temp_input = mat_create(scratch.stk, batch_rows, input_cols);

        memcpy(temp_input->data, input->data, input_rows * input_cols * sizeof(float));

        memset(
            temp_input->data + input_rows * input_cols, 0,
           (batch_rows - input_rows) * input_cols * sizeof(float)
        );

        model->input->value->data = temp_input->data;
        _forward(&model->forward_graph, 0);

        size_t out_size = model->output->c * model->output->h * model->output->w;

        output = mat_create(stk, input_rows, out_size);
        memcpy(
            output->data, model->output->value->data,
            input_rows * out_size * sizeof(float)
        );

        stack_drop_marker(scratch);
    }

    model->input->value->data = checkpoint;

    return output;
}

bool nn_model_save(const nn_model* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        return 0;
    }

    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        const nn_tensor* tensor = model->backward_graph.tensors[i];

        if (tensor->flags & TENSOR_AS_PARAM) {
            size_t data_size = tensor->value->rows * tensor->value->cols;
            if (fwrite(tensor->value->data, sizeof(float), data_size, file) != data_size) {
                fclose(file);
                
                return 0;
            }
        }

        if (tensor->op == TENSOR_OP_BATCHNORM2D) {
            size_t c = tensor->c;
            if (
                fwrite(tensor->running_mean->data, sizeof(float), c, file) != c ||
                fwrite(tensor->running_var->data, sizeof(float), c, file) != c
            ) {
                fclose(file);

                return 0;
            }
        }
    }

    fclose(file);

    return 1;
}

bool nn_model_load(const nn_model* model, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL)  {
        return 0;
    }

    for (size_t i = 0; i < model->backward_graph.tensor_count; i++) {
        nn_tensor* tensor = model->backward_graph.tensors[i];

        if (tensor->flags & TENSOR_AS_PARAM) {
            size_t data_size = tensor->value->rows * tensor->value->cols;

            if (fread(tensor->value->data, sizeof(float), data_size, file) != data_size) {
                fclose(file);

                return 0;
            }
        }

        if (tensor->op == TENSOR_OP_BATCHNORM2D) {
            size_t c = tensor->c;
            if (
                fread(tensor->running_mean->data, sizeof(float), c, file) != c ||
                fread(tensor->running_var->data, sizeof(float), c, file) != c
            ) {
                fclose(file);

                return 0;
            }
        }
    }

    fclose(file);

    return 1;
}