#pragma once


struct THAllocator;


namespace cpptorch
{
    namespace allocator
    {
        THAllocator* get();
        void* generateInfo(long sz);
    }
}
