// Vector Trace (Sum of Elements)
// compile with the following command:
//
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_A;   // host vector
float* h_C;   // host result (partial sums from GPU)
float* d_A;   // device vector
float* d_C;   // device result (partial sums)

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecTrace(const float* A, float* C, int N)
{
    extern __shared__ float cache[];   //  its size is allocated at runtime call

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;  // register for each thread
    while (i < N) {
        temp += A[i];
        i += blockDim.x * gridDim.x;   // go to the next grid 
    }
   
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m
    int ib = blockDim.x / 2;
    while (ib != 0) {
      if (cacheIndex < ib)
      	cache[cacheIndex] += cache[cacheIndex + ib]; 

      __syncthreads();
      ib /= 2;
    }
    
    if(cacheIndex == 0)
      C[blockIdx.x] = cache[0];

}

// Host code
int main(void)
{
    int gid;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU ID: ");
    scanf("%d", &gid);
    printf("%d\n", gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Vector Trace (Sum of Elements)\n");
    int N = 81920007;   // Fixed array size
    printf("Size of the vector: %d\n", N);

    // set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number (2^m) of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        exit(0);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    // blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("%d\n",blocksPerGrid);
    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647!\n");
        exit(0);
    }

    // allocate input vector h_A and result vector h_C in host memory
    int size = N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    h_A = (float*)malloc(size);
    h_C = (float*)malloc(sb);

    // initialize input vector
    RandomInit(h_A, N);

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // allocate vectors in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_C, sb);

    // copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // stop the timer for input
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime(&Intime, start, stop);
    printf("Input time for GPU: %f (ms)\n", Intime);

    // start the timer for computation
    cudaEventRecord(start, 0);

    int sm = threadsPerBlock * sizeof(float);
    VecTrace<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_C, N);

    // stop the timer for computation
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gputime);
    printf("GPU Gflops: %f\n", N / (1000000.0 * gputime));

    // start the timer for output
    cudaEventRecord(start, 0);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    double h_G = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
        h_G += (double)h_C[i];

    // stop the timer for output
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime(&Outime, start, stop);
    printf("Output time for GPU: %f (ms)\n", Outime);

    float gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms)\n", gputime_tot);

    // start the timer for CPU
    cudaEventRecord(start, 0);

    // compute reference solution on CPU
    double h_D = 0.0;
    for (int i = 0; i < N; i++)
        h_D += (double)h_A[i];

    // stop the timer for CPU
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cputime);
    printf("CPU Gflops: %f\n", N / (1000000.0 * cputime));
    printf("Speedup of GPU = %f\n", cputime / gputime_tot);

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result
    printf("Check result:\n");
    double diff = fabs((h_D - h_G) / h_D);
    printf("|(h_G - h_D)/h_D|=%20.15e\n", diff);
    printf("h_G =%20.15e\n", h_G);
    printf("h_D =%20.15e\n", h_D);
    printf("\n");

    free(h_A);
    free(h_C);

    cudaDeviceReset();
}

// Allocates an array with random float entries in (-1,1)
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
//        data[i] = 1.0;   // set all elements to one for checking the code.
}



