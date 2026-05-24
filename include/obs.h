#pragma once

#include <obstack.h> 
#include <stdbool.h>

typedef struct obstack obs;

typedef struct obs_marker {
    obs *stk;
    size_t pos;
} obs_marker;

obs *obs_create();
void obs_destroy(obs *stk);
void *obs_alloc(obs *stk, size_t size, bool clear);
void obs_rollback(obs *stk, size_t pos);

obs_marker obs_get_marker(obs **conflicts, size_t num_conflicts);
void obs_drop_marker(obs_marker scratch);
void obs_destroy_all_markers();