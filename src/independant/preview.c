// preview.c — Vulkan + SDL local preview with CUDA interop
// Uses OPAQUE_FD shared buffer so GPU data is displayed directly
// without any CPU copy.
//
// Usage from cuPlotter:
//   void* pv = createPreview("name", row, col);
//   int fd = previewGetFd(pv);
//   // Import fd into CUDA → device ptr → write GPU data to it
//   updatePreview(pv, d_ptr);  // triggers GPU-to-GPU blit + present

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

struct PreviewContext {
  SDL_Window* window;

  VkInstance instance;
  VkPhysicalDevice physDevice;
  VkDevice device;
  VkQueue queue;
  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;
  VkFormat format;
  VkImage* images;
  uint32_t imgCount;
  VkCommandPool pool;
  VkCommandBuffer* cmdBufs;
  VkFence* fences;
  VkSemaphore acqSem, rendSem;
  uint32_t frame;

  // Shared buffer (OPAQUE_FD export — CUDA writes to it directly)
  VkBuffer sharedBuf;
  VkDeviceMemory sharedMem;
  int sharedFd;
  size_t frameBytes;

  // Intermediate full-size image for buffer→image copy, then blit scales to swapchain
  VkImage srcImage;
  VkDeviceMemory srcMem;
  VkImageView srcImageView;

  // Pre-allocated Vulkan command structures
  VkCommandBufferBeginInfo binfo;
  VkImageMemoryBarrier barSrcDst;   // srcImage → TRANSFER_DST
  VkBufferImageCopy copyRegion;     // buffer → srcImage
  VkImageMemoryBarrier barSrcSrc;   // srcImage → TRANSFER_SRC
  VkImageMemoryBarrier barSwapDst;  // swapchain → TRANSFER_DST (image set per-frame)
  VkImageBlit blit;
  VkImageMemoryBarrier barSwapPres; // swapchain → PRESENT_SRC (image set per-frame)
  VkPipelineStageFlags waitStage;
  VkSubmitInfo submitInfo;
  VkPresentInfoKHR presentInfo;

  int width, height;    // swapchain/window size
  int imgWidth, imgHeight; // full image size (buffer allocation)
  VkColorSpaceKHR colorSpace;
  uint32_t queueFam;
  char active;
};

#define CHECK_VK(call) do { \
  VkResult err_ = (call); \
  if (err_ != VK_SUCCESS && err_ != VK_SUBOPTIMAL_KHR) { \
    fprintf(stderr, "[Preview] VK error line %d: %d\n", __LINE__, (int)err_); \
    exit(1); \
  } \
} while(0)

static int recreateSwapchain(struct PreviewContext* p) {
  vkDeviceWaitIdle(p->device);
  vkDestroySwapchainKHR(p->device, p->swapchain, NULL);

  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p->physDevice, p->surface, &caps);
  VkExtent2D ext = caps.currentExtent;
  if (ext.width == UINT32_MAX) { ext.width = p->width; ext.height = p->height; }
  p->width = ext.width;
  p->height = ext.height;

  VkSwapchainCreateInfoKHR sci = {};
  sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  sci.surface = p->surface;
  sci.minImageCount = 3;
  sci.imageFormat = p->format;
  sci.imageColorSpace = p->colorSpace;
  sci.imageExtent = ext;
  sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sci.queueFamilyIndexCount = 1;
  sci.pQueueFamilyIndices = &p->queueFam;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE;
  if (vkCreateSwapchainKHR(p->device, &sci, NULL, &p->swapchain) != VK_SUCCESS)
    return -1;

  uint32_t newCount;
  vkGetSwapchainImagesKHR(p->device, p->swapchain, &newCount, NULL);
  free(p->images);
  p->images = malloc(newCount * sizeof(VkImage));
  vkGetSwapchainImagesKHR(p->device, p->swapchain, &newCount, p->images);
  p->imgCount = newCount;
  p->frame = 0;

  p->blit.dstOffsets[1].x = p->width;
  p->blit.dstOffsets[1].y = p->height;
  return 0;
}

void* createPreview(const char* windowname, int row, int col){
  struct PreviewContext* p = (struct PreviewContext*)calloc(1, sizeof(struct PreviewContext));
  p->imgWidth = row;
  p->imgHeight = col;
  p->width = row;
  p->height = col;
  p->active = 1;
  p->sharedFd = -1;

  // --- SDL window ---
  assert(SDL_Init(SDL_INIT_VIDEO));
  int winW = row / 8, winH = col / 8;
  if (winW > 480) winW = 480;
  if (winH > 480) winH = 480;
  if (winW < 320) winW = 320;
  if (winH < 320) winH = 320;
  p->window = SDL_CreateWindow(windowname, winW, winH,
    SDL_WINDOW_RESIZABLE);
  assert(p->window != 0);

  // --- Vulkan instance ---
  Uint32 extCount = 0;
  const char* const* extNames = SDL_Vulkan_GetInstanceExtensions(&extCount);
  VkApplicationInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "preview";
  ai.apiVersion = VK_MAKE_VERSION(1, 1, 0);
  VkInstanceCreateInfo ici = {};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  ici.enabledExtensionCount = extCount;
  ici.ppEnabledExtensionNames = extNames;
  CHECK_VK(vkCreateInstance(&ici, NULL, &p->instance));

  // --- Device ---
  uint32_t devN;
  vkEnumeratePhysicalDevices(p->instance, &devN, NULL);
  VkPhysicalDevice* devs = malloc(devN * sizeof(VkPhysicalDevice));
  vkEnumeratePhysicalDevices(p->instance, &devN, devs);
  p->physDevice = devs[0];
  free(devs);

  uint32_t qN;
  vkGetPhysicalDeviceQueueFamilyProperties(p->physDevice, &qN, NULL);
  VkQueueFamilyProperties* qProps = malloc(qN * sizeof(VkQueueFamilyProperties));
  vkGetPhysicalDeviceQueueFamilyProperties(p->physDevice, &qN, qProps);
  uint32_t gfxFam = 0;
  for (uint32_t i = 0; i < qN; i++)
    if (qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { gfxFam = i; break; }
  free(qProps);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo dqci = {};
  dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  dqci.queueFamilyIndex = gfxFam;
  dqci.queueCount = 1;
  dqci.pQueuePriorities = &prio;

  const char* dexts[] = {"VK_KHR_swapchain", "VK_KHR_external_memory", "VK_KHR_external_memory_fd"};
  VkDeviceCreateInfo dci = {};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &dqci;
  dci.enabledExtensionCount = 3;
  dci.ppEnabledExtensionNames = dexts;
  CHECK_VK(vkCreateDevice(p->physDevice, &dci, NULL, &p->device));
  vkGetDeviceQueue(p->device, gfxFam, 0, &p->queue);

  typedef VkResult (*PFN_vkGetMemoryFdKHR)(VkDevice, const VkMemoryGetFdInfoKHR*, int*);
  PFN_vkGetMemoryFdKHR getFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(p->device, "vkGetMemoryFdKHR");
  if (!getFd) { fprintf(stderr, "[Preview] vkGetMemoryFdKHR not available\n"); return NULL; }

  // --- Surface ---
  if (!SDL_Vulkan_CreateSurface(p->window, p->instance, NULL, &p->surface)) {
    fprintf(stderr, "[Preview] CreateSurface: %s\n", SDL_GetError());
    return NULL;
  }

  // --- Format (prefer R8G8B8A8, then B8G8R8A8) ---
  uint32_t fmtN;
  vkGetPhysicalDeviceSurfaceFormatsKHR(p->physDevice, p->surface, &fmtN, NULL);
  VkSurfaceFormatKHR* fmts = malloc(fmtN * sizeof(VkSurfaceFormatKHR));
  vkGetPhysicalDeviceSurfaceFormatsKHR(p->physDevice, p->surface, &fmtN, fmts);
  p->format = fmts[0].format;
  VkColorSpaceKHR cs = fmts[0].colorSpace;
  for (uint32_t i = 0; i < fmtN; i++) {
    if (fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM) { p->format = fmts[i].format; cs = fmts[i].colorSpace; break; }
    if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM) { p->format = fmts[i].format; cs = fmts[i].colorSpace; break; }
  }
  p->colorSpace = cs;
  free(fmts);

  // --- Swapchain ---
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p->physDevice, p->surface, &caps);
  VkExtent2D ext = caps.currentExtent;
  if (ext.width == UINT32_MAX) { ext.width = p->width; ext.height = p->height; }
  p->width = ext.width;
  p->height = ext.height;

  VkSwapchainCreateInfoKHR sci = {};
  sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  sci.surface = p->surface;
  sci.minImageCount = 3;
  sci.imageFormat = p->format;
  sci.imageColorSpace = cs;
  sci.imageExtent = ext;
  sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sci.queueFamilyIndexCount = 1;
  uint32_t qfi = gfxFam;
  sci.pQueueFamilyIndices = &qfi;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE;
  CHECK_VK(vkCreateSwapchainKHR(p->device, &sci, NULL, &p->swapchain));
  p->queueFam = gfxFam;

  vkGetSwapchainImagesKHR(p->device, p->swapchain, &p->imgCount, NULL);
  p->images = malloc(p->imgCount * sizeof(VkImage));
  vkGetSwapchainImagesKHR(p->device, p->swapchain, &p->imgCount, p->images);

  // --- Command pool / buffers / fences / semaphores ---
  VkCommandPoolCreateInfo cpci = {};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = gfxFam;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  CHECK_VK(vkCreateCommandPool(p->device, &cpci, NULL, &p->pool));

  p->cmdBufs = malloc(p->imgCount * sizeof(VkCommandBuffer));
  VkCommandBufferAllocateInfo cbei = {};
  cbei.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbei.commandPool = p->pool;
  cbei.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbei.commandBufferCount = p->imgCount;
  CHECK_VK(vkAllocateCommandBuffers(p->device, &cbei, p->cmdBufs));

  p->fences = malloc(p->imgCount * sizeof(VkFence));
  for (uint32_t i = 0; i < p->imgCount; i++) {
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK_VK(vkCreateFence(p->device, &fci, NULL, &p->fences[i]));
  }

  VkSemaphoreCreateInfo smci = {};
  smci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  CHECK_VK(vkCreateSemaphore(p->device, &smci, NULL, &p->acqSem));
  CHECK_VK(vkCreateSemaphore(p->device, &smci, NULL, &p->rendSem));

  // --- Shared buffer (OPAQUE_FD export) ---
  // 4 bytes per pixel (RGBA/BGRA), allocated at full image size
  p->frameBytes = (size_t)p->imgWidth * p->imgHeight * 4;

  VkBufferCreateInfo bci = {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = p->frameBytes;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  CHECK_VK(vkCreateBuffer(p->device, &bci, NULL, &p->sharedBuf));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(p->device, p->sharedBuf, &memReq);
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(p->physDevice, &memProps);

  // Find memory type that supports OPAQUE_FD
  uint32_t typeIdx = UINT32_MAX;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if (!(memReq.memoryTypeBits & (1u << i))) continue;
    if (!(memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
    VkPhysicalDeviceExternalBufferInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkExternalBufferProperties bprops;
    memset(&bprops, 0, sizeof(bprops));
    bprops.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(p->physDevice, &info, &bprops);
    if ((bprops.externalMemoryProperties.compatibleHandleTypes &
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) != 0) {
      typeIdx = i;
      break;
    }
  }
  if (typeIdx == UINT32_MAX) {
    fprintf(stderr, "[Preview] No memory type supports OPAQUE_FD\n");
    return NULL;
  }

  VkExportMemoryAllocateInfo expInfo = {};
  expInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo mci = {};
  mci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mci.allocationSize = memReq.size;
  mci.memoryTypeIndex = typeIdx;
  mci.pNext = &expInfo;
  CHECK_VK(vkAllocateMemory(p->device, &mci, NULL, &p->sharedMem));
  CHECK_VK(vkBindBufferMemory(p->device, p->sharedBuf, p->sharedMem, 0));

  // Export DMA-BUF FD
  VkMemoryGetFdInfoKHR fdInfo = {};
  fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fdInfo.memory = p->sharedMem;
  fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  CHECK_VK(getFd(p->device, &fdInfo, &p->sharedFd));

  // --- Intermediate image (full image size, for buffer→image then blit→swapchain) ---
  VkImageCreateInfo ici2 = {};
  ici2.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ici2.imageType = VK_IMAGE_TYPE_2D;
  ici2.format = p->format;
  ici2.extent.width = p->imgWidth;
  ici2.extent.height = p->imgHeight;
  ici2.extent.depth = 1;
  ici2.mipLevels = 1;
  ici2.arrayLayers = 1;
  ici2.samples = VK_SAMPLE_COUNT_1_BIT;
  ici2.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici2.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ici2.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  CHECK_VK(vkCreateImage(p->device, &ici2, NULL, &p->srcImage));

  VkMemoryRequirements imgMemReq;
  vkGetImageMemoryRequirements(p->device, p->srcImage, &imgMemReq);
  VkMemoryAllocateInfo imgMci = {};
  imgMci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  imgMci.allocationSize = imgMemReq.size;
  imgMci.memoryTypeIndex = typeIdx;
  CHECK_VK(vkAllocateMemory(p->device, &imgMci, NULL, &p->srcMem));
  CHECK_VK(vkBindImageMemory(p->device, p->srcImage, p->srcMem, 0));

  VkImageViewCreateInfo ivci = {};
  ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ivci.image = p->srcImage;
  ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ivci.format = p->format;
  ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  ivci.subresourceRange.levelCount = 1;
  ivci.subresourceRange.layerCount = 1;
  CHECK_VK(vkCreateImageView(p->device, &ivci, NULL, &p->srcImageView));

  // --- Pre-allocate command structures ---
  memset(&p->binfo, 0, sizeof(p->binfo));
  p->binfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  memset(&p->barSrcDst, 0, sizeof(p->barSrcDst));
  p->barSrcDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  p->barSrcDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  p->barSrcDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  p->barSrcDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSrcDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSrcDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  p->barSrcDst.image = p->srcImage;
  p->barSrcDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->barSrcDst.subresourceRange.levelCount = 1;
  p->barSrcDst.subresourceRange.layerCount = 1;

  memset(&p->copyRegion, 0, sizeof(p->copyRegion));
  p->copyRegion.bufferOffset = 0;
  p->copyRegion.bufferImageHeight = p->imgHeight;
  p->copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->copyRegion.imageSubresource.layerCount = 1;
  p->copyRegion.imageExtent.width = p->imgWidth;
  p->copyRegion.imageExtent.height = p->imgHeight;
  p->copyRegion.imageExtent.depth = 1;

  memset(&p->barSrcSrc, 0, sizeof(p->barSrcSrc));
  p->barSrcSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  p->barSrcSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  p->barSrcSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  p->barSrcSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  p->barSrcSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  p->barSrcSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSrcSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSrcSrc.image = p->srcImage;
  p->barSrcSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->barSrcSrc.subresourceRange.levelCount = 1;
  p->barSrcSrc.subresourceRange.layerCount = 1;

  memset(&p->barSwapDst, 0, sizeof(p->barSwapDst));
  p->barSwapDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  p->barSwapDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  p->barSwapDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  p->barSwapDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSwapDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSwapDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  p->barSwapDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->barSwapDst.subresourceRange.levelCount = 1;
  p->barSwapDst.subresourceRange.layerCount = 1;

  memset(&p->blit, 0, sizeof(p->blit));
  p->blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->blit.srcSubresource.layerCount = 1;
  p->blit.srcOffsets[1].x = p->imgWidth;
  p->blit.srcOffsets[1].y = p->imgHeight;
  p->blit.srcOffsets[1].z = 1;
  p->blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->blit.dstSubresource.layerCount = 1;
  p->blit.dstOffsets[1].x = p->width;
  p->blit.dstOffsets[1].y = p->height;
  p->blit.dstOffsets[1].z = 1;

  memset(&p->barSwapPres, 0, sizeof(p->barSwapPres));
  p->barSwapPres.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  p->barSwapPres.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  p->barSwapPres.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  p->barSwapPres.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  p->barSwapPres.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSwapPres.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  p->barSwapPres.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  p->barSwapPres.subresourceRange.levelCount = 1;
  p->barSwapPres.subresourceRange.layerCount = 1;

  p->waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

  memset(&p->submitInfo, 0, sizeof(p->submitInfo));
  p->submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  p->submitInfo.waitSemaphoreCount = 1;
  p->submitInfo.pWaitSemaphores = &p->acqSem;
  p->submitInfo.pWaitDstStageMask = &p->waitStage;
  p->submitInfo.commandBufferCount = 1;
  p->submitInfo.signalSemaphoreCount = 1;
  p->submitInfo.pSignalSemaphores = &p->rendSem;

  memset(&p->presentInfo, 0, sizeof(p->presentInfo));
  p->presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  p->presentInfo.waitSemaphoreCount = 1;
  p->presentInfo.pWaitSemaphores = &p->rendSem;
  p->presentInfo.swapchainCount = 1;
  p->presentInfo.pSwapchains = &p->swapchain;

  fprintf(stderr, "[Preview] %dx%d %s, %u images, buffer=%zu bytes (img %dx%d), fd=%d\n",
          p->width, p->height,
          p->format == VK_FORMAT_R8G8B8A8_UNORM ? "R8G8B8A8" : "B8G8R8A8",
          p->imgCount, p->frameBytes, p->imgWidth, p->imgHeight, p->sharedFd);
  return p;
}

// Returns the OPAQUE_FD handle for CUDA import
int previewGetFd(void* ptr){
  struct PreviewContext* p = (struct PreviewContext*)ptr;
  return p->sharedFd;
}

// Returns the frame buffer size in bytes
size_t previewGetFrameBytes(void* ptr){
  struct PreviewContext* p = (struct PreviewContext*)ptr;
  return p->frameBytes;
}

// Triggers GPU-to-GPU copy from shared buffer → swapchain + present
// `buffer` is the CUDA device pointer (same physical memory as sharedBuf)
// — it is not read directly; the Vulkan buffer already contains the data.
void updatePreview(void* ptr, void* buffer){
  (void)buffer; // data is already in the shared buffer
  struct PreviewContext* p = (struct PreviewContext*)ptr;
  if (!p->active) return;

  // Poll events
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_EVENT_WINDOW_RESIZED) {
      recreateSwapchain(p);
    } else if (e.type == SDL_EVENT_QUIT || (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_Q)) {
      p->active = 0;
      // Close window + cleanup
      vkDeviceWaitIdle(p->device);
      if (p->sharedFd >= 0) close(p->sharedFd);
      free(p->images);
      for (uint32_t i = 0; i < p->imgCount; i++)
        vkDestroyFence(p->device, p->fences[i], NULL);
      vkDestroySemaphore(p->device, p->acqSem, NULL);
      vkDestroySemaphore(p->device, p->rendSem, NULL);
      vkFreeCommandBuffers(p->device, p->pool, p->imgCount, p->cmdBufs);
      vkDestroyCommandPool(p->device, p->pool, NULL);
      vkDestroySwapchainKHR(p->device, p->swapchain, NULL);
      vkDestroyImageView(p->device, p->srcImageView, NULL);
      vkDestroyImage(p->device, p->srcImage, NULL);
      vkFreeMemory(p->device, p->srcMem, NULL);
      vkDestroyBuffer(p->device, p->sharedBuf, NULL);
      vkFreeMemory(p->device, p->sharedMem, NULL);
      SDL_Vulkan_DestroySurface(p->instance, p->surface, NULL);
      vkDestroyDevice(p->device, NULL);
      vkDestroyInstance(p->instance, NULL);
      if (p->window) SDL_DestroyWindow(p->window);
      SDL_Quit();
      free(p->cmdBufs);
      free(p->fences);
      free(p);
      return;
    }
  }

  // Acquire next image
  uint32_t imgIdx;
  VkResult ar = vkAcquireNextImageKHR(p->device, p->swapchain, 10000000, p->acqSem, NULL, &imgIdx);
  if (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_SUBOPTIMAL_KHR) {
    recreateSwapchain(p);
    return;
  }
  if (ar == VK_TIMEOUT || ar != VK_SUCCESS) return;

  // Wait + reset
  vkWaitForFences(p->device, 1, &p->fences[p->frame], VK_TRUE, 100000000);
  vkResetFences(p->device, 1, &p->fences[p->frame]);
  vkResetCommandBuffer(p->cmdBufs[p->frame], 0);

  // Per-frame: update swapchain image index in pre-allocated structs
  p->barSwapDst.image = p->images[imgIdx];
  p->barSwapPres.image = p->images[imgIdx];

  // Record: buffer → srcImage (1:1) → blit to swapchain (scaled) → present
  vkBeginCommandBuffer(p->cmdBufs[p->frame], &p->binfo);

  vkCmdPipelineBarrier(p->cmdBufs[p->frame], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &p->barSrcDst);
  vkCmdCopyBufferToImage(p->cmdBufs[p->frame], p->sharedBuf, p->srcImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &p->copyRegion);
  vkCmdPipelineBarrier(p->cmdBufs[p->frame], VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &p->barSrcSrc);
  vkCmdPipelineBarrier(p->cmdBufs[p->frame], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &p->barSwapDst);
  vkCmdBlitImage(p->cmdBufs[p->frame],
                 p->srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 p->images[imgIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 1, &p->blit, VK_FILTER_LINEAR);
  vkCmdPipelineBarrier(p->cmdBufs[p->frame], VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &p->barSwapPres);

  vkEndCommandBuffer(p->cmdBufs[p->frame]);

  // Submit + present (update per-frame pointers)
  p->submitInfo.pCommandBuffers = &p->cmdBufs[p->frame];
  vkQueueSubmit(p->queue, 1, &p->submitInfo, p->fences[p->frame]);

  p->presentInfo.pImageIndices = &imgIdx;
  vkQueuePresentKHR(p->queue, &p->presentInfo);

  p->frame = (p->frame + 1) % p->imgCount;
}
