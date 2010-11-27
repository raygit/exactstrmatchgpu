#include <cuda.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

static __constant__ char d_stringPattern[BLOCK_SIZE];

//
// The new stuff here is a simple 'struct' that houses the
// char that i
struct __align__(16) DATA {
 unsigned int match; // initially, it'll be zero and if its 1 it will be the value of the offset within the data
            		 // string
 unsigned int offset; // offset of the character in the provided input data
 char c;     	 // character of the string pointed via (base + offset)
};

//
// The threadId will represent the current shift when this
// function is being executed. Hence, now we need to find a
// match, if possible.
//
// This version uses shared memory
//
template<class TDATA, unsigned int SUBSTRLEN, unsigned int LEN>
__global__ void strstr(const char* substr, TDATA* data) {
    __shared__ TDATA sharedData[BLOCK_SIZE];
 
    int shft = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[threadIdx.x] = *(data + shft); // copy a portion of the text to shared memory for faster access
    __syncthreads();

    const char* s2 = substr;
    unsigned int yes = 1;
    int curr_marker = 0;

    if ( (LEN - shft) < SUBSTRLEN ) {
    	// small optimization: no point writing since by default DATA.match = 0
    	//(sharedData[shft]).match = 0;
        return;
    }
    for( int i = threadIdx.x ; curr_marker <= SUBSTRLEN && i < LEN; curr_marker++, i++ ) {
        if ( s2[curr_marker] && (s2[curr_marker] != (sharedData[i].c) ) ) {
            yes = 0;
            break;
        }
    }
    if ( yes == 1 ) {
       //(sharedData[shft]).match = yes;
    	(*(data + shft)).match = yes;
    }
    
}
//
// This version uses both shared & constant memory
// __constant__ is used to store the pattern string
// __shared__ is used to store the segment of string data
//
template<class TDATA>
__global__ void strstr2(TDATA* data,  int len, int substrlen) {
    __shared__ TDATA* sharedData[BLOCK_SIZE];
 
    int shft = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[threadIdx.x] = data[shft]; // copy a portion of the text to shared memory for faster access
    __syncthreads();

    const char* s2 = d_stringPattern;
    unsigned int yes = 1;
    int curr_marker = 0;

    if ( (len - shft) < substrlen ) {
    	// small optimization: no point writing since by default DATA.match = 0
        //(sharedData[shft]).match = 0;
        return;
    }
    for( int i = threadIdx.x ; curr_marker <= substrlen && i < len; curr_marker++, i++ ) {
        if ( s2[curr_marker] && (s2[curr_marker] != (sharedData[i].c) ) ) {
            yes = 0;
            break;
        }
    }
    if ( yes == 1 ) {
       (sharedData[shft]).match = yes;
    }
}
//
// Read in the given data file and hope it doesn't burst
// the memory limits of the machine or that defined by 'DATA_SIZE'
//
char* readfile(const char* filename) {
    FILE* f;
    char* data = (char*)malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(char));
 
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
void print_shifts(DATA *iptr, int strlen) {
    for(unsigned int i = 0; i < strlen; i++ ) {
        if (iptr[i].match == 1)
            printf("Match found at position: %d\n", iptr[i].offset);
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
    DATA* final_data = (DATA*) malloc( strlen(mainString) * sizeof(DATA) );
    strcpy(subString, argv[3]);
    n = strlen(mainString)+1;
    
    DATA vecd[strlen(mainString)];
    {
		unsigned int len = strlen(mainString);
		for( int i = 0; i < len; i++) {
		DATA d;
		d.offset = i;
		d.c = mainString[i];
		d.match = 0;
		vecd[i] = d;
		}
	}

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
    DATA* d_data = 0;

    cutilSafeCall( cudaMalloc((void**)&d_substr, (strlen(subString)+1)*sizeof(char)) );
    
    cutilSafeCall( cudaMalloc((void**)&d_data, strlen(mainString)*sizeof(DATA)) );

    cutilSafeCall( cudaMemcpy(d_data, vecd, sizeof(DATA)* strlen(mainString), cudaMemcpyHostToDevice ) );
    cutilSafeCall( cudaMemcpy(d_substr, subString, sizeof(char)*(strlen(subString)+1), cudaMemcpyHostToDevice) );
    
    dim3 threadsPerBlocks(BLOCK_SIZE, 1);
    dim3 numBlocks((n/threadsPerBlocks.x) + 1, 1);

	// start timer!
    cudaEventRecord(start_event, 0);

	// conduct actual search!!
	unsigned int SUBSTRLEN = strlen(subString);
	unsigned int LENOFDATA = strlen(mainString);
    strstr<DATA, 2, 1181741><<<numBlocks,threadsPerBlocks>>>(d_substr, d_data);
    //strstr2<<<numBlocks,threadsPerBlocks>>>(d_substr, d_data, strlen(mainString), strlen(subString), d_finalres);

	// copy the data back
	cutilSafeCall( cudaMemcpy( final_data, d_data, sizeof(DATA)*strlen(mainString), cudaMemcpyDeviceToHost) );

	// stop timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize( stop_event );

    cudaEventElapsedTime( &time, start_event, stop_event );
    cudaEventDestroy( start_event ); // cleanup
    cudaEventDestroy( stop_event ); // cleanup

	
    error = cudaGetLastError();
    if ( error ) {
    }
    printf("done and it took: %f milliseconds\n", time);
 
    // check whether the output is correct
    printf("-------------------------------\n");
    print_shifts(final_data, strlen(mainString)+1);
    printf("-------------------------------\n");

    cudaFree(d_substr);
    cudaFree(d_data);
    free(final_data);

    cudaThreadExit();

    return 0;
}
