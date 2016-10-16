#include "allocator.h"
#include "../include/allocator.h"

#include <set>
#include <vector>
#include <assert.h>
#include <TH/TH.h>


struct MemoryAllocation
{
    void *buf;
    long capacity;
};


class MemoryCache
{
public:
    MemoryCache()
    {
        memory_block_info_.reserve(1024);
        free_block_info_index_.reserve(1024);
        available_memory_block_.reserve(1024);
    }

    int requestIndex()
    {
        MemoryAllocation m = {};
        if (!free_block_info_index_.empty())
        {
            int idx = *free_block_info_index_.rbegin();
            free_block_info_index_.pop_back();
            memory_block_info_[idx] = m;
            return idx + 1;
        }
        memory_block_info_.push_back(m);
        return (int)memory_block_info_.size();
    }


    void* alloc(int index, long size)
    {
        MemoryAllocation &m = memory_block_info_[index];
        for (size_t i = 0; i < available_memory_block_.size(); i++)
        {
            MemoryAllocation &findm = available_memory_block_[i];
            if (findm.capacity >= size)
            {
                m.capacity = findm.capacity;
                void *buf = findm.buf;
                available_memory_block_.erase(available_memory_block_.begin() + i);
                return buf;
            }
        }
        // cannot find proper memory block
        m.capacity = size;
        return malloc(size);
    }

    void* re_alloc(int index, void *ptr, long new_size)
    {
        MemoryAllocation &m = memory_block_info_[index];
        for (size_t i = 0; i < available_memory_block_.size(); i++)
        {
            MemoryAllocation &findm = available_memory_block_[i];
            if (findm.capacity >= new_size)
            {
                // cache old
                if (m.capacity > 0)
                {
                    m.buf = ptr;
                    insertIntoAvailableMemory(m);
                }

                // assign new
                m.capacity = findm.capacity;
                void *buf = findm.buf;
                available_memory_block_.erase(available_memory_block_.begin() + i);
                return buf;
            }
        }
        // cannot find proper memory block
        m.capacity = new_size;
        return realloc(ptr, new_size);
    }

    void release(int index, void *buf)
    {
        MemoryAllocation &m = memory_block_info_[index];
        m.buf = buf;
        insertIntoAvailableMemory(m);
        free_block_info_index_.push_back(index);
    }
    
    void cleanup()
    {
        for (auto it = available_memory_block_.begin(); it != available_memory_block_.end(); it++)
        {
            free(it->buf);
        }
        available_memory_block_.clear();
    }

private:
    void insertIntoAvailableMemory(MemoryAllocation ma)
    {
        for (int i = (int)available_memory_block_.size(); i >= 1; i--)
        {
            if (available_memory_block_[i - 1].capacity <= ma.capacity)
            {
                available_memory_block_.insert(available_memory_block_.begin() + i, ma);
                return;
            }
        }
        available_memory_block_.insert(available_memory_block_.begin(), ma);
    }

    // memory block info (include using and unused)
    std::vector<MemoryAllocation> memory_block_info_;

    // empty holes in memory_block_info_
    std::vector<int> free_block_info_index_;

    // memory blocks which can be reused
    std::vector<MemoryAllocation> available_memory_block_;
};

static MemoryCache *cache_ = nullptr;


static void* mallocWrapper(void* ctx, long size)
{
    if (size == 0)
    {
        return nullptr;
    }
    if (cache_ && ctx)
    {
        return cache_->alloc((int)(long long)ctx - 1, size);
    }
    return malloc(size);
}

static void* reallocWrapper(void* ctx, void* ptr, long size)
{
    if (cache_ && ctx)
    {
        return cache_->re_alloc((int)(long long)ctx - 1, ptr, size);
    }
    return realloc(ptr, size);
}

static void freeWrapper(void* ctx, void* ptr)
{
    if (cache_ && ctx)
    {
        cache_->release((int)(long long)ctx - 1, ptr);
    }
    else
    {
        free(ptr);
    }
}

//////////////////////////////////////////////////////////////////////////

void cpptorch::allocator::init()
{
    cache_ = new MemoryCache();
}

void cpptorch::allocator::cleanup()
{
    if (cache_)
    {
        cache_->cleanup();
        delete cache_;
        cache_ = nullptr;
    }
}

//////////////////////////////////////////////////////////////////////////

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

void* cpptorch::allocator::requestIndex()
{
    if (cache_)
    {
        return (void*)(long long)cache_->requestIndex();
    }
    return nullptr;
}
