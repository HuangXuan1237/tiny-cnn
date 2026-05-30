#pragma once

#include <obstack.h> 
#include <stdbool.h>

typedef struct obstack arena;

typedef struct {
    arena *ar;
    size_t pos;
} arena_marker;

arena *arena_create();
void arena_destroy(arena *ar);
void *arena_alloc(arena *ar, size_t size, bool clear);
void arena_rollback(arena *ar, size_t pos);

arena_marker arena_get_marker(arena **conflicts, size_t num_conflicts);
void arena_drop_marker(arena_marker scratch);
void arena_destroy_all_markers();