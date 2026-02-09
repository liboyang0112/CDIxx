void* createVideo(const char* filename, int row, int col, int fps, const char* stream_sock);
void* flushVideo(void* ptr, void* buffer);
void* flushVideo_float(void* ptr, void* buffer);  //buffer is float
void* saveVideo(void* ptr);

