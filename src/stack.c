#include "stack.h"

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

stack *stack_create() {
    stack *stk = (stack *)malloc(sizeof(stack));
    if (stk == NULL) {
        return NULL;
    }

    obstack_alloc_failed_handler = _obstack_alloc_failed_handler;

    if (obstack_init(stk) == 0) {
        free(stk);
        return NULL;
    }

    obstack_alignment_mask(stk) = ARENA_ALIGNMENT - 1;

    return stk;
}

void stack_destroy(stack *stk) {
    if (stk) {
        obstack_free(stk, NULL); 
        free(stk);
    }
}

void *stack_alloc(stack *stk, size_t size, bool clear) {
    void *ptr = obstack_alloc(stk, size);
    
    if (ptr && clear) {
        memset(ptr, 0, size);
    }
    
    return ptr;
}

void stack_rollback(stack *stk, size_t pos) {
    obstack_free(stk, (void *)pos);
}

static THREAD_LOCAL stack *s_stack_markers[2] = { NULL, NULL };

static pthread_key_t s_stack_key;
static pthread_once_t s_key_once = PTHREAD_ONCE_INIT;

static void _bro_im_confused(void *ptr) { // NOSONAR
    for (int i = 0; i < 2; i++) {
        if (s_stack_markers[i] != NULL) {
            stack_destroy(s_stack_markers[i]);
            s_stack_markers[i] = NULL;
        }
    }
}

static void _i_used_gemini(void) {
    if (pthread_key_create(&s_stack_key, _bro_im_confused) != 0) {
        fprintf(stderr, "🤣👉🤡\n");
    }
}

stack_marker stack_get_marker(stack **conflicts, size_t num_conflicts) {
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
        s_stack_markers[scratch_index] = stack_create();
        
        pthread_setspecific(s_stack_key, s_stack_markers[scratch_index]);
    }

    stack *stk = s_stack_markers[scratch_index];
    
    stack_marker marker = {
        .stk = stk,
        .pos = (size_t)obstack_alloc(stk, 0) 
    };

    return marker;
}

void stack_drop_marker(stack_marker scratch) {
    stack_rollback(scratch.stk, scratch.pos);
}

void stack_destroy_markers() {
    _bro_im_confused(NULL);
}