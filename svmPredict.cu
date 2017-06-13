/** File:    svmPredict.cu
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *           use to get accuracy of training result on GPU
 *
 * Compile:  nvcc -o svmPredict svmPredict.cu
 * Run:      ./svmPredict ./data/test-mnist ./data/train-mnist.model 1500 784
 *           ./data/test-mnist: input test data set
 *           ./data/train-mnist.model: input model data
 *           1500       : number of training data 
 *           784        : dimension of feature space
 *
 * Notes:
 *    1.  Need modified_SMO executable to train first in order to get model file. 
 *    2.  The test file shouldn't be the same with the training data.
 *
 * Output: percentage of prediction accuracy of input data
 *
 * Author: Wei-Hsiang Teng
 * History: 2017/6/9       created
 *          2017/6/13      change datatype from float to double
 *	        
 */  

#include <string.h> /* for memset */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> // for estimate elapsed time
#include <math.h>
#define STR_SIZE 8192

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(-1);                                                               \
    }                                                                          \
}

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}
/**
* name:        rbf_kernel
*
* description: kernel generates kernel function K(X_i, X_j) of X which is gaussian.
* input:       x1[]: coordinates of testing data set
*              x2[]: coordinates of support vectors
*              dim: number of dimension of coordinates
*              i, j: index of kernel function K(X_i, X_j)
*              gamma: parameter for guassian kernel: exp(-gamma*|X_i - X_j|^2) 
*
* output:      K(X_i, X_j)      
* 
*/
__device__ double rbf_kernel(double x1[], double x2[], int i, int j, int dim, double gamma)
{
	double ker = 0.0;
	int m;

	for (m = 0; m < dim; m++)
	{
		ker += (x1[i * dim + m] - x2[j * dim + m]) * (x1[i * dim + m] - x2[j * dim + m]);
	}
	ker = exp(-1 * gamma * ker);
	
	return ker;
}

__global__ void svmPredict(double* devX1, double* devX2, int* devY, double* devAlphas, int size, int total_sv, int dim, double gamma, double b, int* num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j, result;
	double dual = 0;
	if (i < size) 
	{
		for (j = 0; j < total_sv; j++) {
			dual +=  devAlphas[j] * rbf_kernel(devX1, devX2, i, j, dim, gamma);
		} 
		dual += b;
		result = 1;
		if (dual < 0)
			result = -1;
		if (result == devY[i])
			atomicAdd(num, 1);
	}
}

void read_data(char* file, double x[], int y[], int size, int dim)
{
	int i;
	char s[STR_SIZE];
	const char* delim = ":";
    char *token;
	int index = 0, pre_index = 0;
	FILE *pFile;
	
	pFile = fopen(file, "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", file);
		exit(-1);
	}
	
	for (i = 0; i < size; i++)
	{
		int cnt = 0;
		fgets(s, sizeof(s), pFile);
	    /* get the first token */
	    token = strtok(s, delim);
	    sscanf(token, "%d %d", &y[i], &index);
	    /* walk through other tokens */
	    while( token != NULL ) 
	    {
			if (cnt == 0) {
				token = strtok(NULL, delim);
			}
			if (index > 0)
				sscanf(token, "%lf %d", &x[i * dim + index - 1], &pre_index);
			index = pre_index;
		    token = strtok(NULL, delim);			
			cnt++;
	    }
	}
	fclose(pFile);
}

void read_model(char* file, double x[], double alphas[], int dim, int total_sv)
{
	FILE *pFile;
	int i;
	char s[STR_SIZE];
	const char* delim = ":";
        char *token;
	int index = 0, pre_index = 0;
	
	pFile = fopen(file, "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", file);
		exit(-1);
	}
	fgets(s, sizeof(s), pFile);
	for (i = 0; i < total_sv; i++)
	{
		int cnt = 0;
		fgets(s, sizeof(s), pFile);
	        /* get the first token */
	        token = strtok(s, delim);
	        sscanf(token, "%lf %d", &alphas[i], &index);
	        /* walk through other tokens */
	        while( token != NULL ) 
	        {
			if (cnt == 0) {
				token = strtok(NULL, delim);
			}
			if (index > 0)
				sscanf(token, "%lf %d", &x[i * dim + index - 1], &pre_index);
			index = pre_index;
		        token = strtok(NULL, delim);			
			cnt++;
	        }
	}
	fclose(pFile);
}

int main(int argc, char* argv[])
{
	int size, dim, total_sv, correct_num;
	double* x1, *x2;
	double gamma, b;
	int *y1;
	double* alphas;
	double start, end;
	
	/* device variables */
	double* devX1;
	double* devX2;
	int* devY;
	double* devAlphas;
	int* devNum;
	
	if (argc < 5) {
		printf("%s data_file model_file data_size data_dim\n", argv[0]);
		exit(-1);
	}
	
	size = atoi(argv[3]);
	dim = atoi(argv[4]);
	
	GET_TIME(start);
	x1 = (double *)malloc(size*dim*sizeof(double));
	memset(x1, 0, sizeof(double)*size*dim);
	y1 = (int *)malloc(size*sizeof(double));
	
	/* read files */
	read_data(argv[1], x1, y1, size, dim);
	/* read model */
	FILE *fp;
	fp = fopen(argv[2], "r");
	if (fp == NULL)
	{
		printf("can't open file %s\n", argv[2]);
		exit(-1);
	}
	fscanf(fp, "%d %lf %lf", &total_sv, &gamma, &b);
	fclose(fp);
	x2 = (double *)malloc(total_sv*dim*sizeof(double));
	memset(x2, 0, sizeof(double)*total_sv*dim);
	alphas = (double *)malloc(total_sv*sizeof(double));
	read_model(argv[2], x2, alphas, dim, total_sv);
	
	/* allocate device memory */
	CHECK(cudaMalloc((void**)&devX1, size * dim * sizeof(double)));
	CHECK(cudaMalloc((void**)&devY, size * sizeof(int)));
	CHECK(cudaMalloc((void**)&devNum, sizeof(int)));
	CHECK(cudaMemcpy(devX1, x1, size * dim * sizeof(double), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(devY, y1, size * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&devX2, total_sv * dim * sizeof(double)));
	CHECK(cudaMalloc((void**)&devAlphas, total_sv * sizeof(double)));
	CHECK(cudaMemcpy(devX2, x2, total_sv * dim * sizeof(double), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(devAlphas, alphas, total_sv * sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 block(32);
	dim3 grid((size + block.x - 1)/block.x);
	
	svmPredict<<<grid, block>>>(devX1, devX2, devY, devAlphas, size, total_sv, dim, gamma, b, devNum);
	CHECK(cudaMemcpy(&correct_num, devNum, sizeof(int), cudaMemcpyDeviceToHost));
	GET_TIME(end);
	printf("accuracy (%d/%d): %1.5f\n", correct_num, size, (double)correct_num/size);
	printf("elapsed time is %lf seconds\n", end - start);
	
	free(y1);
    free(x2);
	free(alphas);
	cudaFree(devX1);
	cudaFree(devX2);
	cudaFree(devY);
	cudaFree(devAlphas);
	cudaFree(devNum);
	return 0;
}
