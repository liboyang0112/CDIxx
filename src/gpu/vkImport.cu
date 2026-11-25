// vkImport.cu — Import Vulkan OPAQUE_FD into CUDA as a device pointer
// Uses CUDA Runtime API (same pattern as vulkan_player/kernel.cu).

#include <cuda_runtime.h>
#include <cuda.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include "cudaDefs_h.cu"

cuFunc(rgbToRGBA, (const uint8_t* rgb, uint8_t* rgba), (rgb, rgba), {
  cuda1Idx();
  rgba[index*4]   = rgb[index*3];
  rgba[index*4+1] = rgb[index*3+1];
  rgba[index*4+2] = rgb[index*3+2];
  rgba[index*4+3] = 255;
})

// Import a Vulkan OPAQUE_FD and return a CUDA device pointer to the memory.
int importVulkanFd(int fd, size_t size, void** dPtr) {
  cudaExternalMemoryHandleDesc desc = {};
  desc.type = (cudaExternalMemoryHandleType)CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  desc.size = size;

  cudaExternalMemory_t extMem;
  if (cudaImportExternalMemory(&extMem, &desc) != cudaSuccess) {
    fprintf(stderr, "[vkImport] cudaImportExternalMemory failed: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    return -1;
  }

  cudaExternalMemoryBufferDesc bufDesc = {};
  bufDesc.offset = 0;
  bufDesc.size = size;
  bufDesc.flags = 0;

  if (cudaExternalMemoryGetMappedBuffer(dPtr, extMem, &bufDesc) != cudaSuccess) {
    fprintf(stderr, "[vkImport] cudaExternalMemoryGetMappedBuffer failed: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    return -1;
  }

  return 0;
}
