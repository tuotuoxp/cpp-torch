#pragma once
#include "../include/allocator.h"


struct THAllocator;


namespace cpptorch
{
    namespace allocator
    {
        API THAllocator* get();
        API void* requestIndex();
    }
}
