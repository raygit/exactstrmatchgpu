/*
 Copyright 2010. All rights reserved 
 Author: Raymond Tay
 version: 1.0
*/
#include <cuda.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

static __constant__ char d_stringPattern[BLOCK_SIZE];

//
// The threadId will represent the current shift when this
// function is being executed. Hence, now we need to find a
// match, if possible.
//
template<class TDATA, unsigned int SUBSTRLEN, unsigned int LEN>
 __global__ void strstr(const char* substr, const char* data,  int len, int substrlen, int* results) {
     int shft = blockIdx.x * blockDim.x + threadIdx.x;
 
     const char* s1 = data;
     const char* s2 = substr;
     unsigned int yes = 1;
     int curr_marker = 0;
 
     if ( (len - shft) < substrlen ) {
         results[shft] = 0;
         return;
     }
     for( int i = shft ; curr_marker <= SUBSTRLEN && i < len; curr_marker++, i++ ) {
             if ( s2[curr_marker] && (s2[curr_marker] != s1[i]) ) {
                 yes = 0;
                 break;
             }
     }
     if ( yes == 1 )
        results[shft] = yes;
 }


//
// The threadId will represent the current shift when this
// function is being executed. Hence, now we need to find a
// match, if possible.
//
// This version uses shared memory
//
template<class TDATA, unsigned int SUBSTRLEN, unsigned int LEN>
__global__ void strstr2(const char* substr, const char* data, int* results) {
    __shared__ char sharedData[BLOCK_SIZE + SUBSTRLEN];
 
    int shft = blockIdx.x * blockDim.x + threadIdx.x;

	if ( threadIdx.x == (warpSize - 1) )
		for(int i = 0; i < SUBSTRLEN; ++i)
			sharedData[threadIdx.x + i] = data[shft+i];
	else 
		sharedData[threadIdx.x] = data[shft]; // copy a portion of the text to shared memory for faster access
    __syncthreads();

    const char* s2 = substr;
    unsigned int yes = 1;
    int curr_marker = 0;

    if ( (LEN - shft) < SUBSTRLEN ) {
        results[shft] = 0;
        return;
    }
    for( int i = threadIdx.x ; curr_marker <= SUBSTRLEN && i < LEN; curr_marker++, i++ ) {
        if ( s2[curr_marker] && (s2[curr_marker] != sharedData[i]) ) {
            yes = 0;
            break;
        }
    }
    if ( yes == 1 ) {
       results[shft] = yes;
    }
}
//
// This version uses both shared & constant memory
// __constant__ is used to store the pattern string
// __shared__ is used to store the segment of string data
//
template<class TDATA, unsigned int SUBSTRLEN, unsigned int LEN>
__global__ void strstr2(const char* data, int* results) {
    __shared__ char sharedData[BLOCK_SIZE + SUBSTRLEN];
 
    int shft = blockIdx.x * blockDim.x + threadIdx.x;

	if ( threadIdx.x == (warpSize - 1) )
		for(int i = 0; i < SUBSTRLEN; ++i)
			sharedData[threadIdx.x + i] = data[shft+i];
	else 
    	sharedData[threadIdx.x] = data[shft]; // copy a portion of the text to shared memory for faster access
    __syncthreads();

    const char* s2 = d_stringPattern;
    unsigned int yes = 1;
    int curr_marker = 0;

    if ( (LEN - shft) < SUBSTRLEN ) {
        results[shft] = 0;
        return;
    }
    for( int i = threadIdx.x ; curr_marker <= SUBSTRLEN && i < LEN; curr_marker++, i++ ) {
        if ( s2[curr_marker] && (s2[curr_marker] != sharedData[i]) ) {
            yes = 0;
            break;
        }
    }
    if ( yes == 1 ) {
       results[shft] = yes;
    }
}
//
// Read in the given data file and hope it doesn't burst
// the memory limits of the machine or that defined by 'DATA_SIZE'
//
char* readfile(const char* filename) {
    FILE* f;
    char* data = (char*)malloc( 1181741 * sizeof(char));
 
    if ( (f = fopen(filename, "r")) != NULL ) {
        // read in the entire file and store into memory
        // hopefully it doesn't exhause the entire RAM on
        // the machine or defy the limits as defined by DATA_SIZE
       fscanf(f, "%s", data);
    }
    fclose(f);
    return data;
}

//
// simple print function to see the shifts in the res array
//
void print_shifts(int *iptr, int strlen) {
    for(unsigned int i = 0; i < strlen; i++ ) {
        if (iptr[i] == 1)
            printf("Match found at position: %d\n", i);
    }
}

int main(int argc, char** argv) {
    int cuda_device = 0; // variable used to denote the device ID
    int n = 0;           // number of ints in the data set
    cudaError_t error;   // capture returned error code
    cudaEvent_t start_event, stop_event; // data structures to capture events in GPU
    float time;

	// Sanity checks
	{
	    // check the compute capability of the device
	    int num_devices=0;
	    cutilSafeCall( cudaGetDeviceCount(&num_devices) );
	    if(0==num_devices)
	    {
	        printf("your system does not have a CUDA capable device\n");
	        return 1;
	    }
    	if( argc > 1 )
       		cuda_device = atoi( argv[1] );

	    // check if the command-line chosen device ID is within range, exit if not
	    if( cuda_device >= num_devices )
	    {
	        printf("choose device ID between 0 and %d\n", num_devices-1);
	        return 1;
	    }

    	cudaSetDevice( cuda_device );

		if ( argc < 4 ) {
      		printf("Usage: bruteforcematcher <device number> <data file> <string pattern>\n");
      		return -1;
    	}
	} // end of sanity checks


    // read in the filename and string pattern to be searched
    char* mainString = readfile( argv[2] );
    char* subString = (char*)malloc( (strlen(argv[3]) + 1) * sizeof(char) );
    strcpy(subString, argv[3]);
    n = strlen(mainString);

    // initializing the GPU timers
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );
   
    cudaDeviceProp deviceProp;
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (1 == deviceProp.major) && (deviceProp.minor < 1))
        printf("%s does not have compute capability 1.1 or later\n", deviceProp.name);

    printf("> Device name : %s\n", deviceProp.name );
    printf("> CUDA Capable SM %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    printf("> Data Size = %d\n", n);
    printf("> String Pattern = %s\n\n", subString);

    // allocate host memory
    char* d_substr = 0;
    char* d_data = 0;
    int*  d_finalres = 0;
    int* finalres = (int*)malloc( (strlen(mainString))*sizeof(int) );

    cutilSafeCall( cudaMalloc((void**)&d_substr, (strlen(subString))*sizeof(char)) );
    
    cutilSafeCall( cudaMemcpyToSymbol(d_stringPattern, subString, sizeof(char)*(strlen(subString)+1)) );
    cutilSafeCall( cudaMalloc((void**)&d_data, (strlen(mainString))*sizeof(char)) );
    cutilSafeCall( cudaMalloc((void**)&d_finalres, (strlen(mainString))*sizeof(int)) );

    cutilSafeCall( cudaMemcpy(d_data, mainString, sizeof(char)*(strlen(mainString)), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy(d_substr, subString, sizeof(char)*(strlen(subString)), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemset(d_finalres, 0, sizeof(int)*strlen(mainString)) );
    
    dim3 threadsPerBlocks(BLOCK_SIZE, 1);
    dim3 numBlocks((int)ceil((float)n/threadsPerBlocks.x), 1);

	printf("Launching kernel with %d blocks, %d threads per block\n", numBlocks.x, threadsPerBlocks.x);
	// start timer!
    cudaEventRecord(start_event, 0);

	// conduct actual search!!
	
	// using no optimizations
    //strstr<<<numBlocks,threadsPerBlocks>>>(d_substr, d_data, strlen(mainString), strlen(subString), d_finalres);
    
    // using both shared & const memory 
    strstr2<char*, 2, 1181741><<<numBlocks,threadsPerBlocks>>>(d_data, d_finalres);
    
    // using only shared memory
    //strstr2<<<numBlocks,threadsPerBlocks>>>(d_substr, d_data, strlen(mainString), strlen(subString), d_finalres);

	// stop timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize( stop_event );

    cudaEventElapsedTime( &time, start_event, stop_event );
    cudaEventDestroy( start_event ); // cleanup
    cudaEventDestroy( stop_event ); // cleanup

    error = cudaGetLastError();
    if ( error ) {
    	printf("Error caught: %s\n", cudaGetErrorString( error ));
    }
    printf("done and it took: %f milliseconds\n", time);

    cutilSafeCall( cudaMemcpy(finalres, d_finalres, (strlen(mainString))*sizeof(int), cudaMemcpyDeviceToHost) );

 
    // check whether the output is correct
    printf("-------------------------------\n");
    print_shifts(finalres, strlen(mainString)+1);
    printf("-------------------------------\n");

    cudaFree(d_substr);
    cudaFree(d_data);
    cudaFree(d_finalres);
    free(finalres);
	free(subString);
	free(mainString);
	
    return 0;
}
