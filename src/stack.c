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

stack_marker stack_get_marker(stack **conflicts, size_t num_conflicts) {
    int scratch_index = -1;

    for (int i = 0; i < 2; i++) {
        bool is_conflicting = 0;
        for (size_t j = 0; j < num_conflicts; j++) {
            if (s_stack_markers[i] == conflicts[j]) {
                is_conflicting = 1;
                break;
            }
        }
        if (is_conflicting == 0) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) {
        return (stack_marker) { 0 };
    }

    stack **selected = &s_stack_markers[scratch_index];

    if (*selected == NULL) {
        *selected = stack_create(); 
    }

    return (stack_marker) {
        .stk = *selected,
        .pos = (size_t)obstack_next_free(*selected) 
    };
}

void stack_drop_marker(stack_marker scratch) {
    stack_rollback(scratch.stk, scratch.pos);
}