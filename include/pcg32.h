#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct pcg32_state {
    uint64_t state;
    uint64_t inc;

    float prev_norm;
} pcg32_state;

void pcg32_set_seed_r(pcg32_state* state, uint64_t init_state, uint64_t init_seq);
void pcg32_set_seed(uint64_t init_state, uint64_t init_seq);

uint32_t pcg32_random_r(pcg32_state* state);
uint32_t pcg32_random();

float pcg32_randomf_r(pcg32_state* state);
float pcg32_randomf();

float pcg32_gaussian_r(pcg32_state* state);
float pcg32_gaussian();