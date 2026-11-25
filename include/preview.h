#include <stddef.h>
#include <stdint.h>

// Vulkan + SDL preview with CUDA interop (OPAQUE_FD)
void* createPreview(const char* windowname, int row, int col);
int previewGetFd(void* ptr);
size_t previewGetFrameBytes(void* ptr);
void updatePreview(void* ptr, void* buffer);

// MJPEG streaming over Unix socket
void* createStream(const char* sockPath, int row, int col);
int streamThrottled(void* ptr);
void updateStream(void* ptr, const unsigned char* jpegData, int jpegSize);
void destroyStream(void* ptr);
