#include    "wb.h"
#define TILE_WIDTH 4
#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    // blockDim.x = blockDim.x = TILE_WIDTH
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // XXX 
    float p_value = 0;
    for (int phase = 0; phase < (numAColumns/TILE_WIDTH) ; phase++) {
        Ads[threadIdx.y][threadIdx.x] = A[row*numAColumns + phase*TILE_WIDTH + threadIdx.x];
        Bds[threadIdx.x][threadIdx.y] = B[(phase*TILE_WIDTH+threadIdx.x)*numBColumns + row];
        //Bds[threadIdx.y][threadIdx.x] = B[row*numBColumns+phase*TILE_WIDTH+threadIdx.x];
        //Ads[threadIdx.x][threadIdx.y] = A[(phase*TILE_WIDTH+threadIdx.x)*numAColumns+row];
        
        //this is wrong answer. iterate it self for check. i learned a lot from iterative checking.
        //Ads[threadIdx.x][threadIdx.y] = A[phase*TILE_WIDTH + col*numAColumns + threadIdx.y];

        __syncthreads(); // synchronize threads theres. all threads finished between x=0,y=0;x=1,y=0;...;x=blockdim-1,y=0;x=0,y=1;x=0,y=2;...;x=0,y=blockdim-1.

        // threads starts parallel again.
        for (int k=0; k<TILE_WIDTH;k++) {
            p_value += Ads[threadIdx.y][k] * Bds[k][threadIdx.x]; // each thread has it own p_value in register.
        __syncthreads();
        }
    }
    C[row*numCColumns + col] = p_value;
  //__shared__ float tileM[TILE_WIDTH][TILE_WIDTH];
  //__shared__ float tileN[TILE_WIDTH][TILE_WIDTH];

  //int bx = blockIdx.x;
  //int by = blockIdx.y;

  //int tx = threadIdx.x;
  //int ty = threadIdx.y;

  //int row = by * TILE_WIDTH + ty;
  //int col = bx * TILE_WIDTH + tx;

  //float result = 0;
  //for(int t = 0; t < numAColumns/TILE_WIDTH; t++) {
  //  tileM[ty][tx] = A[row * numAColumns + t * TILE_WIDTH + tx];
  //  tileN[ty][tx] = B[(t * TILE_WIDTH + ty) * numBColumns + col];

  //  __syncthreads();
  //  for (int k = 0; k < TILE_WIDTH; k++) {
  //    result += tileM[ty][k] * tileN[k][tx];
  //    __syncthreads();
  //  }

  //}

  //C[row * numCColumns + col] = result;

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    wbTime_stop(Generic, "Importing data and creating memory on host");
    hostC = (float*)malloc(sizeof(float)*numCRows*numCColumns);

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc(&deviceA, sizeof(float)*numARows*numAColumns);
    cudaMalloc(&deviceB, sizeof(float)*numBRows*numBColumns);
    cudaMalloc(&deviceC, sizeof(float)*numCRows*numCColumns);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice);
    //cudaMemcpy(deviceC, hostC, sizeof(float)*numCRows*numCColumns, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((numCColumns+block.x-1)/block.x, (numCRows+block.y-1)/block.y);
    
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

