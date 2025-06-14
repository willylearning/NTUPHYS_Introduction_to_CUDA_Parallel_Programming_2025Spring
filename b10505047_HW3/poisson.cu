// Solve the Poisson equation on a 3D lattice with boundary conditions.
//
// compile with the following command:
//
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o laplace laplace.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// field variables
float *h_new; // host field vectors
float *h_old;
float *h_rho; // charge density
float *h_C; // result of diff*diff of each block
float *g_new;
float *d_new; // device field vectors
float *d_old;
float *d_rho;
float *d_C;

int MAX = 10000000;   // maximum iterations
double eps = 1.0e-10; // stopping criterion

double eps0 = 8.85e-12; // vacuum permittivity
double pi = 3.141592653589793;

__global__ void poisson(float *phi_old, float *phi_new, float *rho, float *C, bool flag) 
{
    extern __shared__ float cache[];
    float top, bottom, left, right, front, back;
    float diff;
    int site, xm1, ym1, zm1, xp1, yp1, zp1;

    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;
    int Nz = blockDim.z * gridDim.z;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    int cacheIndex = threadIdx.x + threadIdx.y * blockDim.x +
                     threadIdx.z * blockDim.x * blockDim.y;

    site = x + y * Nx + z * Nx * Ny;

    if ((x == 0) || (x == Nx - 1) || (y == 0) || (y == Ny - 1) || (z == 0) || (z == Nz - 1)) {
        diff = 0.0;
    } else {
        xm1 = site - 1;       // x-1
        xp1 = site + 1;       // x+1
        ym1 = site - Nx;      // y-1
        yp1 = site + Nx;      // y+1
        zm1 = site - Nx * Ny; // z-1
        zp1 = site + Nx * Ny; // z+1
        if (flag) {
            left = phi_old[xm1];
            right = phi_old[xp1];
            back = phi_old[ym1];
            front = phi_old[yp1];
            bottom = phi_old[zm1];
            top = phi_old[zp1];
            phi_new[site] = (top + bottom + left + right + front + back + rho[site]) / 6;
        } else {
            left = phi_new[xm1];
            right = phi_new[xp1];
            back = phi_new[ym1];
            front = phi_new[yp1];
            bottom = phi_new[zm1];
            top = phi_new[zp1];
            phi_old[site] = (top + bottom + left + right + front + back + rho[site]) / 6;
        }
        diff = phi_new[site] - phi_old[site];
    }
    cache[cacheIndex] = diff * diff;
    __syncthreads();

    // perform parallel reduction
    int ib = blockDim.x * blockDim.y * blockDim.z / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib];
        __syncthreads();
        ib /= 2;
    }
    int blockIndex = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    if (cacheIndex == 0)
        C[blockIndex] = cache[0];
}

void getResult(float *g_new, int L) 
{   
    // plot numerical solution of the potential along the body diagonal x = y = z, V(r) as a function of r =sqrt(3)*x 
    // with respect to that of the Coulomb potential V_coulomb(r) = 1 / (4*pi*eps0*r), for L = 8,16,32,64
    FILE *fp = fopen("vr_result.dat", "w");
    for (int x = 0; x < L; x++) { // points on the body diagonal x = y = z
        int site = x + x * L + x * L * L;
        float r = sqrt(3.0f) * fabs(x - L / 2.0f);
        float V_numerical = g_new[site];
        float V_coulomb = 1 / (4 * pi * eps0 * r);
        fprintf(fp, "%.6f %.6e %.6e\n", r, V_numerical, V_coulomb);
    }
    fclose(fp);
}

int main(void) 
{
    int gid;              // GPU_ID
    int iter;
    volatile bool flag;   // to toggle between *_new and *_old
    float cputime;
    float gputime;
    float gputime_tot;
    double flops;
    double error;

    printf("Enter the GPU ID (0/1): ");
    scanf("%d", &gid);
    printf("%d\n", gid);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Select GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Solve Poisson equation on a 3D lattice with boundary conditions\n");

    int L; // lattice size
    printf("Enter the size L of the 3D lattice: ");
    scanf("%d", &L);
    printf("%d\n", L);

    // Set the number of threads (tx, ty, tz) per block
    int tx, ty, tz;
    printf("Enter the number of threads (tx,ty,tz) per block: ");
    scanf("%d %d %d", &tx, &ty, &tz);
    printf("%d %d %d\n", tx, ty, tz);
    if (tx * ty * tz > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(0);
    }
    dim3 threads(tx, ty, tz);

    // The total number of threads in the grid is equal to the total number of lattice sites
    int bx = L / tx;
    if (bx * tx != L) {
        printf("The block size in x is incorrect\n");
        exit(0);
    }
    int by = L / ty;
    if (by * ty != L) {
        printf("The block size in y is incorrect\n");
        exit(0);
    }
    int bz = L / tz;
    if (bz * tz != L) {
        printf("The block size in z is incorrect\n");
        exit(0);
    }
    if ((bx > 65535) || (by > 65535) || (bz > 65535)) {
        printf("The grid size exceeds the limit ! \n");
        exit(0);
    }
    dim3 blocks(bx, by, bz);
    printf("The dimension of the grid is (%d, %d, %d)\n", bx, by, bz);

    int CPU;
    printf("To compute the solution vector with CPU/GPU/both (0/1/2) ? ");
    scanf("%d", &CPU);
    printf("%d\n", CPU);
    fflush(stdout);

    // Allocate field vector h_phi in host memory

    int N = L * L * L;
    int size = N * sizeof(float);
    int sb = bx * by * bz * sizeof(float);
    h_old = (float *)malloc(size);
    h_new = (float *)malloc(size);
    g_new = (float *)malloc(size);
    h_C = (float *)malloc(sb);
    h_rho = (float *)malloc(size);

    memset(h_old, 0, size);
    memset(h_new, 0, size);
    memset(h_rho, 0, size);

    // Initialize the field vector with boundary conditions
    // Initialize the charge density, with point charge q=1 at center (L/2, L/2, L/2)
    h_rho[L / 2 + L * (L / 2) + L * L * (L / 2)] = 1.0 / eps0;
    printf("\n");

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if ((CPU == 1) || (CPU == 2)) {

        // start the timer
        cudaEventRecord(start, 0);

        // Allocate vectors in device memory
        cudaMalloc((void **)&d_new, size);
        cudaMalloc((void **)&d_old, size);
        cudaMalloc((void **)&d_rho, size);
        cudaMalloc((void **)&d_C, sb);

        // Copy vectors from host memory to device memory
        cudaMemcpy(d_new, h_new, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_old, h_old, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rho, h_rho, size, cudaMemcpyHostToDevice);

        // stop the timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float Intime;
        cudaEventElapsedTime(&Intime, start, stop);
        printf("Input time for GPU: %f (ms) \n", Intime);

        // start the timer
        cudaEventRecord(start, 0);

        error = 10 * eps; // any value bigger than eps is OK
        iter = 0;         // counter for iterations
        flag = true;

        int sm = tx * ty * tz * sizeof(float); // size of the shared memory in each block

        while ((error > eps) && (iter < MAX)) {
            poisson<<<blocks, threads, sm>>>(d_old, d_new, d_rho, d_C, flag);
            cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
            error = 0.0;
            for (int i = 0; i < bx * by * bz; i++) {
                error = error + h_C[i];
            }
            error = sqrt(error);
            // printf("error = %.15e\n",error);
            // printf("iteration = %d\n",iter);
            iter++;
            flag = !flag;
        }

        printf("error (GPU) = %.15e\n", error);
        printf("total iterations (GPU) = %d\n", iter);

        // stop the timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&gputime, start, stop);
        printf("Processing time for GPU: %f (ms) \n", gputime);
        flops = 7.0 * (L - 2) * (L - 2) * (L - 2) * iter;
        printf("GPU Gflops: %f\n", flops / (1000000.0 * gputime));

        // Copy result from device memory to host memory

        // start the timer
        cudaEventRecord(start, 0);

        cudaMemcpy(g_new, d_new, size, cudaMemcpyDeviceToHost);

        cudaFree(d_new);
        cudaFree(d_old);
        cudaFree(d_C);

        // stop the timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float Outime;
        cudaEventElapsedTime(&Outime, start, stop);
        printf("Output time for GPU: %f (ms) \n", Outime);

        gputime_tot = Intime + gputime + Outime;
        printf("Total time for GPU: %f (ms) \n", gputime_tot);
        fflush(stdout);

        FILE *outg; // save GPU solution in phi_GPU.dat
        outg = fopen("phi_GPU.dat", "w");

        fprintf(outg, "GPU field configuration:\n");
        for (int k = 0; k < L; k++) {
            for (int j = L - 1; j >= 0; j--) {
                for (int i = 0; i < L; i++) {
                    fprintf(outg, "%.2e ", g_new[i + j * L + k * L * L]);
                }
                fprintf(outg, "\n");
            }
            fprintf(outg, "\n");
        }
        fclose(outg);

        getResult(g_new, L);

        printf("\n");
    }

    if (CPU == 1) { // not to compute the CPU solution
        free(h_new);
        free(h_old);
        free(g_new);
        free(h_C);
        cudaDeviceReset();
        exit(0);
    }

    if ((CPU == 0) || (CPU == 2)) { // to compute the CPU solution

        // start the timer
        cudaEventRecord(start, 0);

        // to compute the reference solution

        error = 10 * eps; // any value bigger than eps
        iter = 0;         // counter for iterations
        flag = true;
        double diff;

        float top, bottom, left, right, front, back;
        int site, xm1, ym1, zm1, xp1, yp1, zp1;

        while ((error > eps) && (iter < MAX)) {
            if (flag) {
                error = 0.0;
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            if (x == 0 || x == L - 1 || y == 0 || y == L - 1 || z == 0 || z == L - 1) {
                            } 
                            else {
                                site = x + y * L + z * L * L;
                                xm1 = site - 1;     // x-1
                                xp1 = site + 1;     // x+1
                                ym1 = site - L;     // y-1
                                yp1 = site + L;     // y+1
                                zm1 = site - L * L; // z-1
                                zp1 = site + L * L; // z+1
                                left = h_old[xm1];
                                right = h_old[xp1];
                                back = h_old[ym1];
                                front = h_old[yp1];
                                bottom = h_old[zm1];
                                top = h_old[zp1];
                                h_new[site] = (top + bottom + left + right + front + back + h_rho[site]) / 6;
                                diff = h_new[site] - h_old[site];
                                error = error + diff * diff;
                            }
                        }
                    }
                }
            } else {
                error = 0.0;
                for (int x = 0; x < L; x++) {
                    for (int y = 0; y < L; y++) {
                        for (int z = 0; z < L; z++) {
                            if (x == 0 || x == L - 1 || y == 0 || y == L - 1 || z == 0 || z == L - 1) {
                            } 
                            else {
                                site = x + y * L + z * L * L;
                                xm1 = site - 1;     // x-1
                                xp1 = site + 1;     // x+1
                                ym1 = site - L;     // y-1
                                yp1 = site + L;     // y+1
                                zm1 = site - L * L; // z-1
                                zp1 = site + L * L; // z+1
                                left = h_new[xm1];
                                right = h_new[xp1];
                                back = h_new[ym1];
                                front = h_new[yp1];
                                bottom = h_new[zm1];
                                top = h_new[zp1];
                                h_old[site] = (top + bottom + left + right + front + back + h_rho[site]) / 6;
                                diff = h_new[site] - h_old[site];
                                error = error + diff * diff;
                            }
                        }
                    }
                }
            }
            flag = !flag;
            iter++;
            error = sqrt(error);
            // printf("error = %.15e\n",error);
            // printf("iteration = %d\n",iter);
            
        } // exit if error < eps

        printf("error (CPU) = %.15e\n", error);
        printf("total iterations (CPU) = %d\n", iter);

        // stop the timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&cputime, start, stop);
        printf("Processing time for CPU: %f (ms) \n", cputime);
        flops = 7.0 * (L - 2) * (L - 2) * (L - 2) * iter;
        printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));

        if (CPU == 2) {
            printf("Speed up of GPU = %f\n", cputime / (gputime_tot));
            fflush(stdout);
        }

        // destroy the timer
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        FILE *outc; // save CPU solution in phi_CPU.dat
        outc = fopen("phi_CPU.dat", "w");

        fprintf(outc, "CPU field configuration:\n");
        for (int k = 0; k < L; k++) {
            for (int j = L - 1; j >= 0; j--) {
                for (int i = 0; i < L; i++) {
                    fprintf(outc, "%.2e ", h_new[i + j * L + k * L * L]);
                }
                fprintf(outc, "\n");
            }
            fprintf(outc, "\n");
        }
        fclose(outc);

        printf("\n");

        free(h_new);
        free(h_old);
        free(g_new);
        free(h_C);
    }

    cudaDeviceReset();
}
