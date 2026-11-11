void* createPreview(const char* windowname, int row, int col);
void updatePreview(void* ptr, void* buffer);

void* createStream(const char* sockPath, int row, int col);
void updateStream(void* ptr, const unsigned char* jpegData, int jpegSize);
void destroyStream(void* ptr);
