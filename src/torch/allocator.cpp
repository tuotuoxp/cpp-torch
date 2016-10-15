#include "allocator.h"

#include <set>
#include <TH/TH.h>


struct MemoryAllocation
{
    void *buf;
    long capacity;
    
    bool operator < (const MemoryAllocation &m) const
    {
        if (capacity < m.capacity)
        {
            return true;
        }
        else if (capacity > m.capacity)
        {
            return false;
        }
        else
        {
            return buf < m.buf;
        }
    }
};


class MemoryCache : protected std::set<MemoryAllocation>
{
public:
    void* alloc(long size, MemoryAllocation *mem)
    {
        for (auto it = begin(); it != end(); it++)
        {
            if (it->capacity >= size)
            {
                mem->capacity = it->capacity;
                void *buf = it->buf;
                erase(it);
                return buf;
            }
        }
        return nullptr;
    }

    void release(MemoryAllocation *mem, void *buf)
    {
        mem->buf = buf;
        insert(*mem);
    }
    
    void cleanup()
    {
        for (auto it = begin(); it != end(); it++)
        {
            free(it->buf);
        }
        clear();
    }
};

static MemoryCache *cache_ = nullptr;


static void* mallocWrapper(void* ctx, long size)
{
    MemoryAllocation *mem = (MemoryAllocation*)ctx;
    if (cache_)
    {
        void *ptr = cache_->alloc(size, mem);
        if (ptr)
        {
            return ptr;
        }
    }
    mem->capacity = size;
    return malloc(size);
}

static void* reallocWrapper(void* ctx, void* ptr, long size)
{
    MemoryAllocation *mem = (MemoryAllocation*)ctx;
    mem->capacity = size;
    return realloc(ptr, size);
}

static void freeWrapper(void* ctx, void* ptr)
{
    MemoryAllocation *mem = (MemoryAllocation*)ctx;
    if (cache_)
    {
        cache_->release(mem, ptr);
    }
    else
    {
        free(ptr);
    }
    delete mem;
}

void cpptorch::allocator::init()
{
    cache_ = new MemoryCache();
}

void cpptorch::allocator::clearup()
{
    cache_->cleanup();
}



THAllocator* cpptorch::allocator::get()
{
    static THAllocator allocator =
    {
        mallocWrapper,
        reallocWrapper,
        freeWrapper
    };
    return &allocator;
}

void* cpptorch::allocator::generateInfo(long sz)
{
    MemoryAllocation *mem = new MemoryAllocation();
    mem->capacity = sz;
    return mem;
}
