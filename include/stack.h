#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <obstack.h> 

typedef struct obstack stack;

typedef struct stack_marker {
    stack *stk;
    size_t pos;
} stack_marker;

stack *stack_create();
void stack_destroy(stack *stk);
void *stack_alloc(stack *stk, size_t size, bool clear);
void stack_rollback(stack *stk, size_t pos);

stack_marker stack_get_marker(stack **conflicts, size_t num_conflicts);
void stack_drop_marker(stack_marker scratch);