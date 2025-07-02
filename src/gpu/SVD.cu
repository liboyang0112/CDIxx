#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaConfig.hpp"
#include "memManager.hpp"
#include <cusolverDn.h>
#include <cuda_runtime.h>

// Enum for SVD status codes
typedef enum {
    SVD_SUCCESS = 0,
    SVD_ERROR_INVALID_ARGUMENT,
    SVD_ERROR_MEMORY_ALLOCATION,
    SVD_ERROR_CUSOLVER,
    SVD_ERROR_INVALID_HANDLE,
    SVD_ERROR_UNKNOWN
} SVDStatus;

// Struct to store SVD-related data
typedef struct {
    int M; // Number of rows in matrix A
    int N; // Number of columns in matrix A
    float *d_U; // Device pointer for U matrix
    float *d_V; // Device pointer for V matrix
    float *d_S; // Device pointer for singular values
    float *work; // Workspace for cuSOLVER
    int lwork; // Size of workspace
    cusolverDnHandle_t handle; // cuSOLVER handle
} SVDContext;

// Internal storage for contexts
#define MAX_CONTEXTS 1024
static SVDContext *context_map[MAX_CONTEXTS] = {NULL};
static int next_handle = 0;

// Global error state (can be replaced with thread-local storage for multi-threading)
static SVDStatus last_error_code = SVD_SUCCESS;
static char last_error_message[256] = "";

// Helper function to set and log errors
void svd_set_error(SVDStatus code, const char *message) {
    last_error_code = code;
    strncpy(last_error_message, message, sizeof(last_error_message) - 1);
    last_error_message[sizeof(last_error_message) - 1] = '\0'; // Ensure null termination
    fprintf(stderr, "Error: %s\n", message);
}

// Function to retrieve the last error message
const char* svd_get_last_error() {
    return last_error_message;
}

// Helper function to validate a handle
SVDContext* validate_handle(int handle) {
    if (handle < 0 || handle >= next_handle || !context_map[handle]) {
        svd_set_error(SVD_ERROR_INVALID_HANDLE, "Invalid SVD handle");
        return NULL;
    }
    return context_map[handle];
}

// Initialize the SVD context and return a handle
int svd_init(int M, int N) {
    // Validate dimensions directly here
    if (M <= 0 || N <= 0) {
        svd_set_error(SVD_ERROR_INVALID_ARGUMENT, "Invalid matrix dimensions");
        return -1;
    }

    if (next_handle >= MAX_CONTEXTS) {
        svd_set_error(SVD_ERROR_MEMORY_ALLOCATION, "Maximum number of SVD contexts reached");
        return -1;
    }

    // Allocate a new context
    SVDContext *ctx = (SVDContext*)malloc(sizeof(SVDContext));
    if (!ctx) {
        svd_set_error(SVD_ERROR_MEMORY_ALLOCATION, "Failed to allocate SVDContext");
        return -1;
    }
    ctx->M = M;
    ctx->N = N;

    // Create cuSOLVER handle
    if (cusolverDnCreate(&ctx->handle) != CUSOLVER_STATUS_SUCCESS) {
        free(ctx);
        svd_set_error(SVD_ERROR_CUSOLVER, "Failed to create cuSOLVER handle");
        return -1;
    }

    // Allocate device memory for U, V, and S
    if (cudaMalloc((void**)&ctx->d_U, M * M * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&ctx->d_V, N * N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&ctx->d_S, N * sizeof(float)) != cudaSuccess) {
        cusolverDnDestroy(ctx->handle);
        free(ctx);
        svd_set_error(SVD_ERROR_MEMORY_ALLOCATION, "Failed to allocate device memory");
        return -1;
    }

    // Query workspace size
    if (cusolverDnSgesvd_bufferSize(ctx->handle, M, N, &ctx->lwork) != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroy(ctx->handle);
        cudaFree(ctx->d_U);
        cudaFree(ctx->d_V);
        cudaFree(ctx->d_S);
        free(ctx);
        svd_set_error(SVD_ERROR_CUSOLVER, "Failed to query workspace size");
        return -1;
    }

    if (cudaMalloc((void**)&ctx->work, ctx->lwork * sizeof(float)) != cudaSuccess) {
        cusolverDnDestroy(ctx->handle);
        cudaFree(ctx->d_U);
        cudaFree(ctx->d_V);
        cudaFree(ctx->d_S);
        free(ctx);
        svd_set_error(SVD_ERROR_MEMORY_ALLOCATION, "Failed to allocate workspace memory");
        return -1;
    }

    // Store the context and return its handle
    int handle = next_handle++;
    context_map[handle] = ctx;
    return handle;
}

// Perform SVD on the input matrix A (already on device)
int svd_execute(int handle, const float *d_A_input) {
    SVDContext *ctx = validate_handle(handle);
    if (!ctx) {
        return SVD_ERROR_INVALID_HANDLE;
    }

    if (!d_A_input) {
        svd_set_error(SVD_ERROR_INVALID_ARGUMENT, "Invalid input matrix");
        return SVD_ERROR_INVALID_ARGUMENT;
    }

    // Perform SVD
    int info = 0;
    cusolverStatus_t status = cusolverDnSgesvd(ctx->handle, 'A', 'A', ctx->M, ctx->N, 
                                               (float*)d_A_input, ctx->M, // Use the input d_A directly
                                               ctx->d_S, ctx->d_U, ctx->M, ctx->d_V, ctx->N,
                                               ctx->work, ctx->lwork, NULL, &info);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        svd_set_error(SVD_ERROR_CUSOLVER, "cuSOLVER SVD computation failed");
        return SVD_ERROR_CUSOLVER;
    }

    if (info != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "SVD failed with info = %d", info);
        svd_set_error(SVD_ERROR_CUSOLVER, msg);
        return SVD_ERROR_CUSOLVER;
    }

    return SVD_SUCCESS;
}

// Free all allocated resources for the given handle
int svd_destroy(int handle) {
    SVDContext *ctx = validate_handle(handle);
    if (!ctx) {
        return SVD_ERROR_INVALID_HANDLE;
    }

    // Free device memory
    cudaFree(ctx->d_U);
    cudaFree(ctx->d_V);
    cudaFree(ctx->d_S);
    cudaFree(ctx->work);

    // Destroy cuSOLVER handle
    cusolverDnDestroy(ctx->handle);

    // Free the context itself
    free(ctx);

    // Remove the context from the map
    context_map[handle] = NULL;

    return SVD_SUCCESS;
}

// Get the device pointer for singular values (d_S)
float* svd_get_singular_values(int handle) {
    SVDContext *ctx = validate_handle(handle);
    if (!ctx) {
        return NULL;
    }
    return ctx->d_S;
}

// Get the device pointer for U matrix (d_U)
float* svd_get_U_matrix(int handle) {
    SVDContext *ctx = validate_handle(handle);
    if (!ctx) {
        return NULL;
    }
    return ctx->d_U;
}

// Get the device pointer for V matrix (d_V)
float* svd_get_V_matrix(int handle) {
    SVDContext *ctx = validate_handle(handle);
    if (!ctx) {
        return NULL;
    }
    return ctx->d_V;
}

// Example usage
void example_usage() {
    // Define matrix A
    const int M = 5;
    const int N = 3;
    float A[M][N] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0}
    };

    // Allocate device memory for A
    float *d_A;
    if (cudaMalloc((void**)&d_A, M * N * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A\n");
        return;
    }
    if (cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy A to device\n");
        cudaFree(d_A);
        return;
    }

    // Initialize SVD context and get handle
    int handle = svd_init(M, N);
    if (handle == -1) {
        fprintf(stderr, "Failed to initialize SVD context: %s\n", svd_get_last_error());
        cudaFree(d_A);
        return;
    }

    // Perform SVD
    int status = svd_execute(handle, d_A);
    if (status != SVD_SUCCESS) {
        fprintf(stderr, "SVD execution failed: %s\n", svd_get_last_error());
    } else {
        // Retrieve device pointers for d_S, d_U, d_V
        float *d_S = svd_get_singular_values(handle);
        float *d_U = svd_get_U_matrix(handle);
        float *d_V = svd_get_V_matrix(handle);

        if (!d_S || !d_U || !d_V) {
            fprintf(stderr, "Failed to retrieve SVD results: %s\n", svd_get_last_error());
        } else {
            // Copy singular values back to host
            float singular[N];
            if (cudaMemcpy(singular, d_S, N * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) {
                printf("Singular values:\n");
                for (int i = 0; i < N; i++) {
                    printf("%f\n", singular[i]);
                }
            } else {
                fprintf(stderr, "Failed to copy singular values to host\n");
            }
        }
    }

    // Clean up
    cudaFree(d_A);
    svd_destroy(handle);
}

int main() {
    example_usage();
    return 0;
}
