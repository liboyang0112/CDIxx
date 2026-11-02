#include <string.h>
#include <stdio.h>
#ifdef NVIMGCODEC_FOUND
#include "nvimgcodec.h"
#include <cuda_runtime.h>

static nvimgcodecInstance_t g_nvc_inst = NULL;

extern "C" int writeJPEG_GPU(void* d_rgb, int height, int width, const char* fname, int quality) {
  if (!g_nvc_inst) {
    nvimgcodecInstanceCreateInfo_t ci;
    memset(&ci, 0, sizeof(ci));
    ci.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.struct_size = sizeof(ci);
    ci.load_builtin_modules = 1;
    ci.load_extension_modules = 1;
    ci.extension_modules_path = "/opt/nvimgcodec/nvimgcodec-linux-x86_64-0.9.0.20_cuda13-archive/lib/extensions";
    nvimgcodecStatus_t st = nvimgcodecInstanceCreate(&g_nvc_inst, &ci);
    if (st != NVIMGCODEC_STATUS_SUCCESS) {
      fprintf(stderr, "writeJPEG_GPU: instance create failed: %d\n", st);
      return -1;
    }
  }
  nvimgcodecImageInfo_t ii;
  memset(&ii, 0, sizeof(ii));
  ii.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
  ii.struct_size = sizeof(ii);
  ii.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
  ii.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
  ii.num_planes = 1;
  ii.plane_info[0].width = width;
  ii.plane_info[0].height = height;
  ii.plane_info[0].row_stride = width * 3;
  ii.plane_info[0].num_channels = 3;
  ii.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  ii.plane_info[0].precision = 8;
  ii.buffer = d_rgb;
  ii.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

  nvimgcodecImage_t img = NULL;
  nvimgcodecStatus_t st = nvimgcodecImageCreate(g_nvc_inst, &img, &ii);
  if (st != NVIMGCODEC_STATUS_SUCCESS) {
    fprintf(stderr, "writeJPEG_GPU: image create failed: %d\n", st);
    return -1;
  }
  nvimgcodecImageInfo_t out_ii = ii;
  strcpy(out_ii.codec_name, "jpeg");
  nvimgcodecCodeStream_t stream = NULL;
  st = nvimgcodecCodeStreamCreateToFile(g_nvc_inst, &stream, fname, &out_ii);
  if (st != NVIMGCODEC_STATUS_SUCCESS) {
    fprintf(stderr, "writeJPEG_GPU: code stream create failed: %d\n", st);
    nvimgcodecImageDestroy(img);
    return -1;
  }
  nvimgcodecExecutionParams_t xp;
  memset(&xp, 0, sizeof(xp));
  xp.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
  xp.struct_size = sizeof(xp);
  xp.device_id = NVIMGCODEC_DEVICE_CURRENT;
  nvimgcodecEncoder_t enc = NULL;
  st = nvimgcodecEncoderCreate(g_nvc_inst, &enc, &xp, NULL);
  if (st != NVIMGCODEC_STATUS_SUCCESS) {
    fprintf(stderr, "writeJPEG_GPU: encoder create failed: %d\n", st);
    nvimgcodecCodeStreamDestroy(stream);
    nvimgcodecImageDestroy(img);
    return -1;
  }
  nvimgcodecEncodeParams_t ep;
  memset(&ep, 0, sizeof(ep));
  ep.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
  ep.struct_size = sizeof(ep);
  ep.quality_type = NVIMGCODEC_QUALITY_TYPE_QUALITY;
  ep.quality_value = quality;
  nvimgcodecFuture_t future = NULL;
  st = nvimgcodecEncoderEncode(enc, &img, &stream, 1, &ep, &future);
  if (st != NVIMGCODEC_STATUS_SUCCESS) {
    fprintf(stderr, "writeJPEG_GPU: encode failed: %d\n", st);
    nvimgcodecEncoderDestroy(enc);
    nvimgcodecCodeStreamDestroy(stream);
    nvimgcodecImageDestroy(img);
    return -1;
  }
  st = nvimgcodecFutureWaitForAll(future);
  nvimgcodecProcessingStatus_t pstatus;
  size_t pstatus_size = 0;
  nvimgcodecFutureGetProcessingStatus(future, &pstatus, &pstatus_size);
  cudaDeviceSynchronize();
  if (pstatus != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
    fprintf(stderr, "writeJPEG_GPU: processing status: 0x%x\n", (unsigned)pstatus);
  }
  nvimgcodecFutureDestroy(future);
  nvimgcodecEncoderDestroy(enc);
  nvimgcodecCodeStreamDestroy(stream);
  nvimgcodecImageDestroy(img);
  return 0;
}
#endif
