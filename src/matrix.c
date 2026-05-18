#include "matrix.h"

matrix *mat_create(stack *stk, size_t rows, size_t cols) {
    matrix *mat = (matrix*)stack_alloc(stk, sizeof(matrix), 1);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)stack_alloc(stk, sizeof(float) * rows * cols, 1);

    mat_fill(mat, 0.0f);

    return mat;
}

bool mat_load(matrix *mat, size_t rows, size_t cols, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        return 0;
    }

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);

    size = size > sizeof(float) * rows * cols ? sizeof(float) * rows * cols : size;

    if (fread(mat->data, 1, size, file) != size) {
        fclose(file);

        return 0;
    }

    fclose(file);

    return 1;
}

void mat_fill(matrix *mat, float value) {
    if (mat == NULL || mat->data == NULL) {
        return; 
    }

    size_t size = mat->rows * mat->cols;

    if (value == 0.0f) {
        memset(mat->data, 0, sizeof(float) * size);
        return;
    }

    for (size_t i = 0; i < size; i++) {
        mat->data[i] = value;
    }
}

bool mat_add(matrix *result, const matrix *x, const matrix *y) { //NOSONAR
    if (result->cols != x->cols) {
        return 0;
    }
    size_t rows = result->rows;
    size_t cols = result->cols;

    if (x->rows == rows && y->rows == rows && x->cols == cols && y->cols == cols) {
        if (result != x) {
            cblas_scopy((int)(rows * cols), x->data, 1, result->data, 1);
        }

        cblas_saxpy((int)(rows * cols), 1.0f, y->data, 1, result->data, 1);

        return 1;
    } else if (x->rows == rows && y->rows == 1 && x->cols == cols && y->cols == cols) {
        for (size_t i = 0; i < rows; i++) {
            float *dst_row = result->data + i * cols;
            const float *a_row = x->data + i * cols;
            if (result != x) {
                cblas_scopy((int)cols, a_row, 1, dst_row, 1);
            }

            cblas_saxpy((int)cols, 1.0f, y->data, 1, dst_row, 1);
        }

        return 1;
    } else if (x->rows == rows && y->rows == 1 && cols == x->cols && cols % y->cols == 0) {
        size_t repeat = cols / y->cols;
        for (size_t i = 0; i < rows; i++) {
            float *dst_row = result->data + i * cols;
            const float *a_row = x->data + i * cols;

            if (result != x) {
                memcpy(dst_row, a_row, cols * sizeof(float));
            }

            for (size_t j = 0; j < y->cols; j++) {
                float val = y->data[j];
                for (size_t r = 0; r < repeat; r++) { //NOSONAR
                    dst_row[j * repeat + r] += val;
                }
            }
        }

        return 1;
    } else if (y->rows == rows && x->rows == 1 && cols == y->cols && cols % x->cols == 0) {
        size_t repeat = cols / x->cols;
        for (size_t i = 0; i < rows; i++) {
            float *dst_row = result->data + i * cols;
            const float *b_row = y->data + i * cols;

            if (result != y) {
                memcpy(dst_row, b_row, cols * sizeof(float));
            }

            for (size_t j = 0; j < x->cols; j++) {
                float val = x->data[j];
                for (size_t r = 0; r < repeat; r++) { //NOSONAR
                    dst_row[j * repeat + r] += val;
                }
            }
        }
        
        return 1;
    }

    return 0;
}

bool mat_argmax(const matrix *mat, size_t *result) {
    if (mat == NULL) {
        return 0;
    }

    size_t rows = mat->rows;
    size_t cols = mat->cols;

    for (size_t i = 0; i < rows; i++) {
        const float *row = mat->data + i * cols;

        size_t max_idx = 0;
        for (size_t j = 1; j < cols; j++) {
            if (row[j] > row[max_idx]) {
                max_idx = j;
            }
        }

        result[i] = max_idx;
    }

    return 1;
}

bool mat_batchnorm2d( //NOSONAR
    matrix *output, const matrix *input, const matrix *gamma, const matrix *beta,
    matrix *running_mean, matrix *running_var,
    float eps, float momentum, bool is_training,
    matrix *x_hat, matrix *inv_std,
    size_t channels, size_t spatial
) {
    size_t batch_size = input->rows;
    float ns = (float)(batch_size * spatial);

    if (is_training) {
        // 使用项目的内存池（stack）替代 __builtin_alloca，保证多线程下的栈安全
        stack_marker scratch = stack_get_marker(NULL, 0);
        float *mean = (float*)stack_alloc(scratch.stk, channels * sizeof(float), 1);
        float *var = (float*)stack_alloc(scratch.stk, channels * sizeof(float), 1);

        for (size_t c = 0; c < channels; c++) {
            mean[c] = 0.0f;
            var[c] = 0.0f;
        }

        for (size_t n = 0; n < batch_size; n++) {
            for (size_t c = 0; c < channels; c++) {
                size_t offset = n * channels * spatial + c * spatial;
                float sum = 0.0f;
                for (size_t s = 0; s < spatial; s++) { // NOSONAR
                    sum += input->data[offset + s];
                }
                mean[c] += sum;
            }
        }

        for (size_t c = 0; c < channels; c++) {
            mean[c] /= ns;
        }

        for (size_t n = 0; n < batch_size; n++) {
            for (size_t c = 0; c < channels; c++) {
                size_t offset = n * channels * spatial + c * spatial;
                float m = mean[c];
                float sum_sq = 0.0f;
                for (size_t s = 0; s < spatial; s++) { // NOSONAR
                    float diff = input->data[offset + s] - m;
                    sum_sq += diff * diff;
                }
                var[c] += sum_sq;
            }
        }
        
        for (size_t c = 0; c < channels; c++) {
            var[c] /= ns;
        }

        for (size_t c = 0; c < channels; c++) {
            running_mean->data[c] = momentum * running_mean->data[c] + (1.0f - momentum) * mean[c];
            running_var->data[c]  = momentum * running_var->data[c] + (1.0f - momentum) * var[c];
            
            inv_std->data[c] = 1.0f / sqrtf(var[c] + eps);
        }

        #pragma omp parallel for collapse(2)
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t c = 0; c < channels; c++) {
                size_t offset = n * channels * spatial + c * spatial;
                float m = mean[c];
                float inv_s = inv_std->data[c];
                float g = gamma->data[c];
                float b = beta->data[c];

                float scale = inv_s * g;
                float bias = b - m * scale;

                for (size_t s = 0; s < spatial; s++) { // NOSONAR
                    size_t idx = offset + s;
                    float x_val = input->data[idx];
                    
                    float x_hat_val = (x_val - m) * inv_s;
                    x_hat->data[idx] = x_hat_val; 
                    
                    output->data[idx] = x_val * scale + bias;
                }
            }
        }

        // 释放临时内存，回滚状态
        stack_drop_marker(scratch);
    } else {
        #pragma omp parallel for collapse(2)
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t c = 0; c < channels; c++) {
                size_t offset = n * channels * spatial + c * spatial;
                float rm = running_mean->data[c];
                float rv = running_var->data[c];
                float g = gamma->data[c];
                float b = beta->data[c];

                float inv_s = 1.0f / sqrtf(rv + eps);
                float scale = inv_s * g;
                float bias = b - rm * scale;

                for (size_t s = 0; s < spatial; s++) { // NOSONAR
                    output->data[offset + s] = input->data[offset + s] * scale + bias;
                }
            }
        }
    }

    return 1;
}

bool mat_grad_batchnorm2d( //NOSONAR
    matrix *grad_input, matrix *grad_gamma, matrix *grad_beta,
    const matrix *input, const matrix *gamma, 
    const matrix *x_hat, const matrix *inv_std,
    const matrix *grad_output,
    size_t channels, size_t spatial
) {
    size_t batch_size = input->rows;
    float ns = (float)(batch_size * spatial);

    // 使用项目的内存池（stack）替代 __builtin_alloca
    stack_marker scratch = stack_get_marker(NULL, 0);
    float *sum_dy = (float*)stack_alloc(scratch.stk, channels * sizeof(float), 1);
    float *sum_dy_x_hat = (float*)stack_alloc(scratch.stk, channels * sizeof(float), 1);

    for (size_t c = 0; c < channels; c++) {
        sum_dy[c] = 0.0f;
        sum_dy_x_hat[c] = 0.0f;
    }

    for (size_t n = 0; n < batch_size; n++) {
        for (size_t c = 0; c < channels; c++) {
            size_t offset = n * channels * spatial + c * spatial;
            float s_dy = 0.0f;
            float s_dy_x_hat = 0.0f;
            
            for (size_t s = 0; s < spatial; s++) {
                size_t idx = offset + s;
                float dy = grad_output->data[idx];
                
                s_dy += dy;
                s_dy_x_hat += dy * x_hat->data[idx];
            }

            sum_dy[c] += s_dy;
            sum_dy_x_hat[c] += s_dy_x_hat;
        }
    }

    for (size_t c = 0; c < channels; c++) {
        grad_gamma->data[c] += sum_dy_x_hat[c];
        grad_beta->data[c]  += sum_dy[c];
    }

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t c = 0; c < channels; c++) {
            size_t offset = n * channels * spatial + c * spatial;
            
            float g = gamma->data[c];
            float std = inv_std->data[c];
            float s_dy = sum_dy[c];
            float s_dy_x_hat = sum_dy_x_hat[c];
            
            float factor = (g * std) / ns;

            for (size_t s = 0; s < spatial; s++) {
                size_t idx = offset + s;
                float dy = grad_output->data[idx];

                grad_input->data[idx] += factor * (ns * dy - s_dy - x_hat->data[idx] * s_dy_x_hat);
            }
        }
    }

    // 释放临时内存，回滚状态
    stack_drop_marker(scratch);

    return 1;
}

static void __matrix_img2col( //NOSONAR
    matrix *result, const matrix *input,
    size_t in_c, size_t in_h, size_t in_w, size_t k_size,
    size_t stride, size_t padding
) {
    size_t out_h = (in_h + 2 * padding - k_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_size) / stride + 1;
    size_t out_hw = out_h * out_w;

    memset(result->data, 0, sizeof(float) * result->rows * result->cols);

    #pragma omp parallel for collapse(3)
    for (size_t c = 0; c < in_c; ++c) {
        for (size_t ky = 0; ky < k_size; ++ky) {
            for (size_t kx = 0; kx < k_size; ++kx) {
                size_t row_idx = (c * k_size * k_size + ky * k_size + kx) * out_hw;
                for (size_t y = 0; y < out_h; ++y) { //NOSONAR
                    int iy = (int)(y * stride + ky - padding);
                    if (iy < 0 || iy >= (int)in_h) {
                        continue;
                    }

                    const float *in_ptr = &input->data[c * in_h * in_w + iy * in_w];
                    float *res_ptr = &result->data[row_idx + y * out_w];

                    for (size_t x = 0; x < out_w; ++x) {
                        int ix = (int)(x * stride + kx - padding);
                        if (ix >= 0 && ix < (int)in_w) {
                            res_ptr[x] = in_ptr[ix];
                        }
                    }
                }
            }
        }
    }
}

bool mat_conv2d( //NOSONAR
    matrix *result, matrix *input, matrix *kernel, 
    size_t in_c, size_t in_h, size_t in_w, 
    size_t k_size, size_t stride, size_t padding
) {
    size_t batch_size = input->rows;
    size_t out_h = (in_h + 2 * padding - k_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_size) / stride + 1;
    size_t out_c = kernel->rows * kernel->cols / (in_c * k_size * k_size);

    if (result->rows != batch_size || result->cols != out_c * out_h * out_w) {
        return 0;
    }

    stack_marker scratch = stack_get_marker(NULL, 0);
    
    matrix *col_mat = mat_create(scratch.stk, in_c * k_size * k_size, out_h * out_w);
    matrix *temp = mat_create(scratch.stk, out_c, out_h * out_w);

    size_t in_stride = in_c * in_h * in_w;
    size_t out_stride = out_c * out_h * out_w;

    for (size_t b = 0; b < batch_size; b++) {
        matrix input_view = {
            .rows = 1, 
            .cols = in_stride, 
            .data = input->data + b * in_stride
        };

        __matrix_img2col(
            col_mat, &input_view, in_c, in_h, in_w, k_size, stride, padding
        );

        matrix kernel_view = {
            .rows = out_c,
            .cols = in_c * k_size * k_size,
            .data = kernel->data
        };
        
        mat_mul(temp, &kernel_view, col_mat, 0.0f, 0, 0);

        memcpy(result->data + b * out_stride, temp->data, sizeof(float) * out_stride);
    }

    stack_drop_marker(scratch);

    return 1;
}

static void __matrix_col2img( //NOSONAR
    matrix *result, const matrix *col_mat,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride, size_t padding
) {
    size_t out_h = (in_h + 2 * padding - k_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_size) / stride + 1;

    for (size_t c = 0; c < in_c; c++) {
        for (size_t ky = 0; ky < k_size; ky++) {
            for (size_t kx = 0; kx < k_size; kx++) {
                size_t col_row = c * k_size * k_size + ky * k_size + kx;
                for (size_t y = 0; y < out_h; y++) { //NOSONAR
                    for (size_t x = 0; x < out_w; x++) {
                        int iy = (int)(y * stride + ky - padding);
                        int ix = (int)(x * stride + kx - padding);
                        
                        if (iy >= 0 && iy < (int)in_h && ix >= 0 && ix < (int)in_w) {
                            result->data[
                                c * in_h * in_w + iy * in_w + ix
                            ] += col_mat->data[
                                col_row * (out_h * out_w) + (y * out_w + x)
                            ];
                        }
                    }
                }
            }
        }
    }
}

bool mat_grad_conv2d( //NOSONAR
    matrix *input_grad, matrix *kernel_grad,
    matrix *input, matrix *kernel, matrix *grad,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride, size_t padding
) {
    size_t out_h = (in_h + 2 * padding - k_size) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_size) / stride + 1;
    size_t single_filter_elements = in_c * k_size * k_size;
    size_t out_c = kernel->rows * kernel->cols / single_filter_elements;
    size_t batch_size = input->rows;

    stack_marker scratch = stack_get_marker(NULL, 0);
    matrix *col_mat = mat_create(scratch.stk, single_filter_elements, out_h * out_w);
    matrix *d_col = mat_create(scratch.stk, single_filter_elements, out_h * out_w);

    matrix k_reshaped = { .rows = out_c, .cols = single_filter_elements, .data = kernel->data };
    matrix kg_reshaped;
    if (kernel_grad) {
        kg_reshaped.rows = out_c;
        kg_reshaped.cols = single_filter_elements;
        kg_reshaped.data = kernel_grad->data;
    }

    for (size_t b = 0; b < batch_size; b++) {
        matrix input_view = {
            .rows = 1, .cols = in_c * in_h * in_w,
            .data = input->data + b * in_c * in_h * in_w
        };
        matrix grad_view = {
            .rows = 1, .cols = out_c * out_h * out_w,
            .data = grad->data + b * out_c * out_h * out_w
        };
        matrix g_reshaped = {
            .rows = out_c, .cols = out_h * out_w, .data = grad_view.data
        };

        __matrix_img2col(col_mat, &input_view, in_c, in_h, in_w, k_size, stride, padding);

        if (kernel_grad) {
            mat_mul(&kg_reshaped, &g_reshaped, col_mat, 1.0f, 0, 1);
        }

        if (input_grad) {
            mat_mul(d_col, &k_reshaped, &g_reshaped, 0.0f, 1, 0);

            matrix img_view = {
                .rows = 1, .cols = in_c * in_h * in_w,
                .data = input_grad->data + b * in_c * in_h * in_w
            };

            __matrix_col2img(
                &img_view, d_col, in_c, in_h, in_w, k_size, stride, padding
            );
        }
    }
    
    stack_drop_marker(scratch);

    return 1;
}

bool mat_cross_entropy(matrix *result, const matrix *pred, const matrix *target) {
    if (pred->rows != target->rows || pred->cols != target->cols) {
        return 0;
    }
    if (result->rows != pred->rows || result->cols != 1) return 0;

    for (size_t i = 0; i < pred->rows; i++) {
        float val = 0.0f;
        const float *pred_row = pred->data + i * pred->cols;
        const float *target_row = target->data + i * target->cols;

        for (size_t j = 0; j < pred->cols; j++) {
            if (target_row[j] > 0.0f) {
                val -= target_row[j] * logf(pred_row[j] + 1e-10f);
            }
        }
        result->data[i] = val;
    }
    return 1;
}

bool mat_grad_cross_entropy(
    matrix *pred_grad, matrix *target_grad,
    const matrix *pred, const matrix *target,
    const matrix *grad
) {
    if (pred->rows != target->rows || pred->cols != target->cols) return 0;
    if (grad->rows != pred->rows || grad->cols != 1) return 0;

    size_t batch_size = pred->rows;
    size_t n = pred->cols;

    for (size_t b = 0; b < batch_size; b++) {
        const float *pred_row = pred->data + b * n;
        const float *target_row = target->data + b * n;
        float grad_val = grad->data[b];

        if (pred_grad != NULL) {
            float *pg_row = pred_grad->data + b * n;
            for (size_t i = 0; i < n; i++) {
                pg_row[i] += (-target_row[i] / (pred_row[i] + 1e-10f)) * grad_val;
            }
        }

        if (target_grad != NULL) {
            float *tg_row = target_grad->data + b * n;
            for (size_t i = 0; i < n; i++) {
                tg_row[i] += -logf(pred_row[i] + 1e-10f) * grad_val;
            }
        }
    }
    return 1;
}


bool mat_gavgpool2d(
    matrix *output, const matrix *input, size_t c, size_t h, size_t w
) {
    if (output->rows != input->rows || output->cols != c) {
        return 0;
    }

    size_t batch_size = input->rows;
    size_t spatial = h * w;
    float inv_spatial = 1.0f / (float)spatial;

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ch = 0; ch < c; ch++) {
            float sum = 0.0f;
            size_t base = b * c * spatial + ch * spatial;

            for (size_t p = 0; p < spatial; p++) {
                sum += input->data[base+p];
            }

            output->data[b*c+ch] = sum * inv_spatial;
        }
    }

    return 1;
}

bool mat_grad_gavgpool2d(
    matrix *grad_input, const matrix *grad_output, size_t c, size_t h, size_t w
) {
    if (grad_input->rows != grad_output->rows || grad_input->cols != c * h * w) {
        return 0;
    }

    size_t batch_size = grad_output->rows;
    size_t spatial = h * w;
    float inv_spatial = 1.0f / (float)spatial;

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ch = 0; ch < c; ch++) {

            float g = grad_output->data[b*c+ch] * inv_spatial;
            size_t base = b * c * spatial + ch * spatial;

            for (size_t p = 0; p < spatial; p++) {
                grad_input->data[base+p] += g;
            }
        }
    }

    return 1;
}

bool mat_maxpool2d( //NOSONAR
    matrix *result, const matrix *input,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride
) {
    size_t out_h = (in_h - k_size) / stride + 1;
    size_t out_w = (in_w - k_size) / stride + 1;
    size_t out_c = in_c;
    size_t batch_size = input->rows;
    if (result->rows != batch_size || result->cols != out_c * out_h * out_w) {
        return 0;
    }

    size_t in_stride = in_c * in_h * in_w;
    size_t out_stride = out_c * out_h * out_w;

    for (size_t b = 0; b < batch_size; b++) {
        const float *in_data = input->data + b * in_stride;
        float *out_data = result->data + b * out_stride;

        for (size_t c = 0; c < out_c; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) { //NOSONAR
                    float max_val = -FLT_MAX;
                    for (size_t kh = 0; kh < k_size; kh++) {
                        for (size_t kw = 0; kw < k_size; kw++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            float val = in_data[c * (in_h * in_w) + ih * in_w + iw];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }

                    out_data[c * (out_h * out_w) + oh * out_w + ow] = max_val;
                }
            }
        }
    }

    return 1;
}

bool mat_grad_maxpool2d( //NOSONAR
    matrix *result, const matrix *input, const matrix *grad,
    size_t in_c, size_t in_h, size_t in_w,
    size_t k_size, size_t stride
) {
    size_t out_h = (in_h - k_size) / stride + 1;
    size_t out_w = (in_w - k_size) / stride + 1;

    size_t batch_size = input->rows;
    size_t in_stride = in_c * in_h * in_w;
    size_t out_stride = in_c * out_h * out_w;

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < in_c; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) { //NOSONAR
                    float max_v = -FLT_MAX;
                    size_t max_ih = 0, max_iw = 0; //NOSONAR
                    
                    for (size_t kh = 0; kh < k_size; kh++) {
                        for (size_t kw = 0; kw < k_size; kw++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            float v = input->data[b * in_stride + c * in_h * in_w + ih * in_w + iw];
                            if (v > max_v) {
                                max_v = v;
                                max_ih = ih;
                                max_iw = iw;
                            }
                        }
                    }
                    
                    result->data[b * in_stride + c * in_h * in_w + max_ih * in_w + max_iw] += 
                        grad->data[b * out_stride + c * out_h * out_w + oh * out_w + ow];
                }
            }
        }
    }
    return 1;
}


bool mat_mul(
    matrix *result, const matrix *x, const matrix *y,
    float beta, bool transpose_x, bool transpose_y
) {
    size_t x_rows = transpose_x ? x->cols : x->rows;
    size_t x_cols = transpose_x ? x->rows : x->cols;
    
    size_t y_rows = transpose_y ? y->cols : y->rows;
    size_t y_cols = transpose_y ? y->rows : y->cols;
    
    if (x_cols != y_rows) {
        return 0;
    }

    if (result->rows != x_rows || result->cols != y_cols) {
        return 0;
    }

    CBLAS_TRANSPOSE transA = transpose_x ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = transpose_y ? CblasTrans : CblasNoTrans;
    int M = (int)x_rows;
    int N = (int)y_cols;
    int K = (int)x_cols;
    float alpha = 1.0f;

    cblas_sgemm(
        CblasRowMajor, transA, transB,
        M, N, K, alpha,
        x->data, (int)x->cols,
        y->data, (int)y->cols,
        beta,
        result->data, (int)result->cols
    );

    return 1;
}

bool mat_relu(matrix *result, const matrix *input) {
    if (result->rows != input->rows || result->cols != input->cols) {
        return 0;
    }

    size_t size = input->rows * input->cols;

    for (size_t i = 0; i < size; i++) {
        result->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    }

    return 1;
}

bool mat_grad_relu(
    matrix *result, const matrix *input, const matrix *grad
) {
    if (result->rows != input->rows || result->cols != input->cols) {
        return 0;
    }

    if (result->rows != grad->rows || result->cols != grad->cols) {
        return 0;
    }

    size_t size = result->rows * result->cols;

    
    for (size_t i = 0; i < size; i++) {
        result->data[i] += input->data[i] > 0 ? grad->data[i] : 0;
    }

    return 1;
}

static float *s_ones = NULL;
static size_t s_ones_len = 0;

static void ensure_ones(size_t len) {
    if (s_ones_len < len) {
        s_ones = realloc(s_ones, sizeof(float) * len);
        for (size_t i = 0; i < len; i++) s_ones[i] = 1.0f;
        s_ones_len = len;
    }
}

bool mat_softmax(matrix *result, const matrix *input) {
    if (result->rows != input->rows || result->cols != input->cols) {
        return 0;
    }

    size_t rows = input->rows;
    size_t cols = input->cols;
    ensure_ones(cols);
    
    for (size_t i = 0; i < rows; i++) {
        const float *in_row = input->data + i * cols;
        float *out_row = result->data + i * cols;

        float max_val = in_row[0];
        for (size_t j = 1; j < cols; j++) {
            if (in_row[j] > max_val) max_val = in_row[j];
        }

        for (size_t j = 0; j < cols; j++) {
            out_row[j] = expf(in_row[j] - max_val);
        }

        float sum = cblas_sdot((int)cols, out_row, 1, s_ones, 1);
        float inv_sum = 1.0f / (sum + 1e-10f);
        cblas_sscal((int)cols, inv_sum, out_row, 1);
    }
    return 1;
}

bool mat_grad_softmax(matrix *result, const matrix *input, const matrix *grad) {
    size_t batch = input->rows;
    size_t n = input->cols;

    #pragma omp parallel for
    for (size_t b = 0; b < batch; b++) {
        const float *y_row = input->data + b * n;
        const float *grad_row = grad->data + b * n;
        float *res_row = result->data + b * n;

        float dot = 0.0f;
        for (size_t i = 0; i < n; i++) dot += grad_row[i] * y_row[i];

        for (size_t i = 0; i < n; i++) {
            res_row[i] += y_row[i] * (grad_row[i] - dot);
        }
    }

    return 1;
}

float mat_sum(const matrix *mat) {
    size_t size = mat->rows * mat->cols;
    ensure_ones(size);

    return cblas_sdot((int)size, mat->data, 1, s_ones, 1);
}

bool mat_scale(matrix *mat, float scalar) {
    if (mat == NULL) {
        return 0;
    }

    size_t size = mat->rows * mat->cols;

    cblas_sscal((int)size, scalar, mat->data, 1);

    return 1;
}