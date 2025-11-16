#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include "op/flashattn/flash-nano.cuh"
// #include "op/gemm/gemm_wmma.cuh"
// #include "op/test/shfl.cuh"
// #include "op/test/smem.cuh"
// #include "op/transpose/transpose.cuh"



/*



 */

int main(int argc, char **argv)
{
    const int DEVICE_ID = 0;
    cudaSetDevice(DEVICE_ID);

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, DEVICE_ID);
    std::cout << "DEVICE  ID: " << DEVICE_ID << "\t" << deviceProp.name << std::endl; //
    std::cout << "        SM: " << deviceProp.multiProcessorCount << std::endl; // 128
    std::cout << "threads SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl; // 1536 = 256 * 6
    std::cout << "Shared Mem: " << deviceProp.sharedMemPerBlock / 1024 << " x 1024" << std::endl;  // 48 * 1024
    std::cout << "regs Block: " << deviceProp.regsPerBlock << std::endl; // 65536 = 256 * 256

    std::cout << "--- --- --- --- --- --- --- --- ---" << std::endl;


    flashattn_nano();

    // gemm_wmma_main();

    // test_shfl_main();
    // test_smem_main();

    // transpose_main();


    return 0;
}

