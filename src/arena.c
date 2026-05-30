#include "arena.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define ARENA_ALIGNMENT 64 

#define THREAD_LOCAL __thread

#define obstack_chunk_alloc     malloc
#define obstack_chunk_free      free

static void _obstack_alloc_failed_handler(void) {
    fprintf(stderr, "🤣👉🤡\n");
    exit(EXIT_FAILURE); 
}

arena *arena_create() {
    arena *ar = (arena *)malloc(sizeof(arena));
    if (ar == NULL) {
        return NULL;
    }

    obstack_alloc_failed_handler = _obstack_alloc_failed_handler;

    if (obstack_init(ar) == 0) {
        free(ar);
        return NULL;
    }

    obstack_alignment_mask(ar) = ARENA_ALIGNMENT - 1;

    return ar;
}

void arena_destroy(arena *ar) {
    if (ar) {
        obstack_free(ar, NULL); 
        free(ar);
    }
}

void *arena_alloc(arena *ar, size_t size, bool clear) {
    void *ptr = obstack_alloc(ar, size);
    
    if (ptr && clear) {
        memset(ptr, 0, size);
    }
    
    return ptr;
}

void arena_rollback(arena *ar, size_t pos) {
    obstack_free(ar, (void *)pos);
}

static THREAD_LOCAL arena *s_stack_markers[2] = { NULL, NULL };

static pthread_key_t s_stack_key;
static pthread_once_t s_key_once = PTHREAD_ONCE_INIT;

static void _bro_i_am_confused(void *ptr) { // NOSONAR
    for (int i = 0; i < 2; i++) {
        if (s_stack_markers[i] != NULL) {
            arena_destroy(s_stack_markers[i]);
            s_stack_markers[i] = NULL;
        }
    }
}

static void _i_used_gemini(void) {
    if (pthread_key_create(&s_stack_key, _bro_i_am_confused) != 0) {
        fprintf(stderr, "🤣👉🤡\n");
    }
}

arena_marker arena_get_marker(arena **conflicts, size_t num_conflicts) {
    pthread_once(&s_key_once, _i_used_gemini);

    int scratch_index = -1;

    for (int i = 0; i < 2; i++) {
        bool is_conflicting = 0;
        for (size_t j = 0; j < num_conflicts; j++) {
            if (s_stack_markers[i] == conflicts[j]) {
                is_conflicting = 1;
                break;
            }
        }

        if (!is_conflicting) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) {
        fprintf(stderr, "🤣👉🤡\n");
        exit(EXIT_FAILURE);
    }

    if (s_stack_markers[scratch_index] == NULL) {
        s_stack_markers[scratch_index] = arena_create();
        
        pthread_setspecific(s_stack_key, s_stack_markers[scratch_index]);
    }

    arena *ar = s_stack_markers[scratch_index];
    
    arena_marker marker = {
        .ar = ar,
        .pos = (size_t)obstack_alloc(ar, 0) 
    };

    return marker;
}

void arena_drop_marker(arena_marker scratch) {
    arena_rollback(scratch.ar, scratch.pos);
}

void arena_destroy_all_markers() {
    _bro_i_am_confused(NULL);
}