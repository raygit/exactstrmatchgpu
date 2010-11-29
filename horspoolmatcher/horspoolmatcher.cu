/*
 Copyright 2010. All rights reserved
 Author: Raymond Tay
 version: 1.0
*/
#include <cuda.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

#define ASIZE 340
#define DATA_SIZE 10240000

__device__ int shifts[ASIZE];
__device__ int results[DATA_SIZE];

//
// This is will compute the bad-character shift function
//
__global__ void processPattern(char* x ,int m, int shifts[]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    if ( idx >= m - 1 ) return;
   
    char c = x[idx];
    for( int i = m - 2; i >= idx; --i ) {
        if ( x[i] == c ) {// match is found
            shifts[c] = m - i - 1;
            return;
        }
    }
}

__global__ void search(char *x, int m, char* y, int n, int shifts[], int indx[]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    if ( idx > (n - m) ) return;
    if ( indx[idx] != idx ) return;
   
    unsigned int yes = 1;
    char c = y[idx + m - 1];
    for( int i = 0; i < m - 1; ++i ) {
        // try to match the string
        if ( x[m - 1] != c || x[i] != y[idx + i] ) {
            yes = 0;
            break;
        }
    }
    results[idx] = yes;
}

void precomputeShiftIndx(char* y, int n, int m, int shifts[], int indx[]) {
    int j = 0;
    int limit = n - m;
 
    while (j <= limit ) {
        j += shifts[ y[j + m - 1] ];
        indx[j] = j;
    }
}


//
// Read in the given data file and hope it doesn't burst
// the memory limits of the machine
//
char* readfile(const char* filename) {
    using namespace std;
    string line;
    char* data = (char*)malloc( DATA_SIZE * sizeof(char));
    ifstream myfile (filename);
    if (myfile.is_open())
    {
        while (! myfile.eof() )
        {
            getline (myfile,line);
            strcat(data, line.c_str());
        }
        myfile.close();
    }
    else cout << "Unable to open file";
    return data;
}

void display_results(int n, int  results[]) {
    for( int i =0; i < n; ++i )
        if ( results[i] == 1 )
            printf("Found match at %d\n", i);
}

int main(int argc, char* argv[]) {
    int cuda_device = 0;
    size_t n = 0; // length of main string
    size_t m = 0; // length of substring

    if ( argc < 4 ) {
        printf("Usage: horspoolmatcher <device number> <string to be searched> <string>\n");
        return -1;
    }

    if( argc > 1 )
        cuda_device = atoi( argv[1] );

    //
    // Read in the 'pattern' to be matched against the string in the data
    //
    char* mainString = readfile(argv[3]);
    char* subString = (char*) malloc( (strlen(argv[2]) + 1) * sizeof(char) );
    strcpy(subString, argv[2]);
    n = strlen(mainString);
    m = strlen(subString);
    //
    // Initialize the shift and index array
    //
    int* l_shifts = (int*)malloc( ASIZE * sizeof(int) );
    for( int i = 0; i < ASIZE; ++i )
        l_shifts[i] = m;
    int* l_indx = (int*) malloc( n * sizeof(int) );
    for( int i = 0; i < n; ++i )
        l_indx[i] = -1;
   
    cudaError_t error;
    cudaEvent_t start_event, stop_event;
    float time;
    float time2;
   
    // initializing the GPU timers
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );

    // check the compute capability of the device
    int num_devices=0;
    cutilSafeCall( cudaGetDeviceCount(&num_devices) );
    if(0==num_devices)
    {
        printf("your system does not have a CUDA capable device\n");
        return 1;
    }

    // check if the command-line chosen device ID is within range, exit if not
    if( cuda_device >= num_devices )
    {
        printf("choose device ID between 0 and %d\n", num_devices-1);
        return 1;
    }

    cudaSetDevice( cuda_device );
    cudaDeviceProp deviceProp;

    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (1 == deviceProp.major) && (deviceProp.minor < 1))
    printf("%s does not have compute capability 1.1 or later\n", deviceProp.name);

    printf("> Device name : %s\n", deviceProp.name );
    printf("> CUDA Capable SM %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    printf("> array_size   = %d\n", n);
    //printf("> main string = %s\n", mainString);
    //printf("> sub string = %s\n\n", subString);

    //
    // Allocate global memory to host the pattern, text and other supporting data
    // structures
    //
    char* d_substr = 0;
    int* d_shifts = 0;
    int* d_indx = 0;
    char* d_text = 0;
    char* d_results = 0;
    cudaGetSymbolAddress((void**)&d_shifts, "shifts");
    cutilSafeCall( cudaMalloc((void**)&d_indx, n * sizeof(int)) );
    cutilSafeCall( cudaMalloc((void**)&d_substr, (m + 1)*sizeof(char)) );
    cutilSafeCall( cudaMalloc((void**)&d_text, (strlen(mainString)+1)*sizeof(char)) );
    cutilSafeCall( cudaMemcpy(d_shifts, l_shifts, sizeof(int) * ASIZE, cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy(d_text, mainString, sizeof(char)*(strlen(mainString)+1), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy(d_substr, subString, sizeof(char)*(strlen(subString)+1), cudaMemcpyHostToDevice) );
    error = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(error));
   
    //
    // Pre-process the pattern to be matched
    //
    dim3 threadsPerBlocks(ASIZE, 1);
    int t = m / threadsPerBlocks.x;
    int t1 = m % threadsPerBlocks.x;
    if ( t1 != 0 ) t += 1;
    dim3 numBlocks(t, 1);

    printf("Launching kernel with blocks=%d, threadsperblock=%d\n", numBlocks.x, threadsPerBlocks.x);
    cudaEventRecord(start_event, 0);
    processPattern<<<numBlocks,threadsPerBlocks>>>(d_substr, m, d_shifts);
    cudaThreadSynchronize();

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize( stop_event );
    cudaEventElapsedTime( &time, start_event, stop_event );

/*
cutilSafeCall( cudaMemcpy(l_shifts, d_shifts, sizeof(int) * ASIZE, cudaMemcpyDeviceToHost ) );
for( int i = 0; i < ASIZE; ++i )
    printf("%d, l_shifts[%d]\n",l_shifts[i], i);
*/   

    //
    // Transfer the pre-computed shift indexes from host to device memory
    //
    cutilSafeCall( cudaMemcpy(l_shifts, d_shifts, ASIZE * sizeof(int), cudaMemcpyDeviceToHost) );
    precomputeShiftIndx(mainString , n, m, l_shifts, l_indx);
    cutilSafeCall( cudaMemcpy(d_shifts, l_shifts, ASIZE * sizeof(int), cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(d_indx, l_indx, n * sizeof(int), cudaMemcpyHostToDevice) );
   
    //
    // Perform the actual search
    //
    t = n / threadsPerBlocks.x;
    t1 = n % threadsPerBlocks.x;
    if ( t1 != 0 ) t += 1;
    dim3 numBlocks2(t, 1);
    printf("Launching kernel with blocks=%d, threadsperblock=%d\n", numBlocks2.x, threadsPerBlocks.x);
    cudaEventRecord(start_event, 0);
    search<<<numBlocks2,threadsPerBlocks>>>(d_substr, m, d_text, n, d_shifts, d_indx);
    cudaThreadSynchronize();

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize( stop_event );
    cudaEventElapsedTime( &time2, start_event, stop_event );
  
    cudaEventDestroy( start_event ); // cleanup
    cudaEventDestroy( stop_event ); // cleanup
    printf("done and it took: %f+%f=%f milliseconds\n",time, time2, time+time2);

    cudaGetSymbolAddress((void**)&d_results, "results");
    int* l_results = (int*) malloc( n * sizeof(int) );
    cutilSafeCall( cudaMemcpy(l_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost) );
    display_results(n, l_results);
       
 cudaFree(d_substr);
 cudaFree(d_shifts);
 cudaFree(d_indx);
 cudaFree(d_text);
 free(mainString);
 free(subString);
 free(l_indx);
 free(l_shifts);
 free(l_results);
 
cudaThreadExit();

return 0;
}
