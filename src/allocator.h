#pragma once


struct THAllocator;


namespace cpptorch
{
    namespace allocator
    {
        THAllocator* get();
        void* requestIndex();
    }
}
