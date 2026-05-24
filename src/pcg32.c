#include "pcg32.h"

#include <math.h>
#include <stdbool.h>

#define PI 3.14159265358979323846F
#define PCG32_INV_24BIT 5.9604644775390625E-8F

static pcg32_state s_pcg32_state = {
    .state = 0X853C49E6748FEA9BULL,
    .inc = 0XDA3E39CB94B95BDBULL
};

void pcg32_set_seed_r(pcg32_state* state, uint64_t init_state, uint64_t init_seq) {
    state->state = 0U;
    state->inc = (init_seq << 1U) | 1U;
    pcg32_random_r(state);

    state->state += init_state;
    pcg32_random_r(state);

    state->prev_norm = NAN;
}

void pcg32_set_seed(uint64_t init_state, uint64_t init_seq) {
    pcg32_set_seed_r(&s_pcg32_state, init_state, init_seq);
}

uint32_t pcg32_random_r(pcg32_state* state) {
    uint64_t prev_state = state->state;
    state->state = prev_state * 6364136223846793005ULL + state->inc;

    uint32_t xorshifted = (uint32_t)(((prev_state >> 18U) ^ prev_state) >> 27U);
    uint32_t rot = (uint32_t)(prev_state >> 59U);

    uint32_t result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31)); //NOSONAR

    return result;
}

uint32_t pcg32_random() {
    return pcg32_random_r(&s_pcg32_state);
}

float pcg32_randomf_r(pcg32_state* state) {
    return (pcg32_random_r(state) >> 8) * PCG32_INV_24BIT;
}

float pcg32_randomf() {
    return pcg32_randomf_r(&s_pcg32_state);
}

float pcg32_gaussian_r(pcg32_state* state) {
    if (isnan(state->prev_norm) == 0) {
        float result = state->prev_norm;
        state->prev_norm = NAN;

        return result;
    }

    float u1 = 0.0F;

    do {
        u1 = pcg32_randomf_r(state);
    } while ( u1 == 0.0F);
    
    float u2 = pcg32_randomf_r(state);

    float mag = sqrtf(-2.0F * logf(u1));

    float z0 = mag * cosf(2.0F * PI * u2);
    float z1 = mag * sinf(2.0F * PI * u2);

    state->prev_norm = z1;

    return z0;
}

float pcg32_gaussian() {
    return pcg32_gaussian_r(&s_pcg32_state);
}