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
    THCudaInit(global_thc);
}

void cpptorch::cuda::free()
{
    THCudaShutdown(global_thc);
    THCState_free(global_thc);
    global_thc = nullptr;
}
