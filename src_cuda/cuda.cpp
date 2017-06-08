#include "../include/cpptorch_cuda.h"

#include <THC/THC.h>


static THCState *global_thc = nullptr;


THCState* getCudaState()
{
    return global_thc;
}


void cpptorch::cuda::init()
{
    global_thc = THCState_alloc();
    /* Enable the caching allocator unless THC_CACHING_ALLOCATOR=0 */
    char* thc_caching_allocator = getenv("THC_CACHING_ALLOCATOR");
    if (!thc_caching_allocator || strcmp(thc_caching_allocator, "0") != 0) {
        THCState_setDeviceAllocator(global_thc, THCCachingAllocator_get());
        global_thc->cudaHostAllocator = &THCCachingHostAllocator;
    }
    THCudaInit(global_thc);
    #ifdef USE_MAGMA
      THCMagma_init(state);
    #endif
}

void cpptorch::cuda::free()
{
    THCudaShutdown(global_thc);
    THCState_free(global_thc);
    //global_thc = nullptr;
}
