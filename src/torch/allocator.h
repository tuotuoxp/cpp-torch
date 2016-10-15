#pragma once


struct THAllocator;


namespace cpptorch
{
    namespace allocator
    {
        void init();
        void clearup();

        THAllocator* get();
        void* generateInfo(long sz);
    }
}
