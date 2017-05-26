/** File:     mat_modified_SMO.cu
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *			 the CUDA version which pre-calculates kernel function.
 *
 * Compile:  nvcc -o mat_modified_SMO mat_modified_SMO.cu
 * Run:      ex: ./mat_modified_SMO ./data/train-mnist ./data/train-mnist.model 10000 784 1 0.01 0.001
 *           ./data/train-mnist: input training data set
 *           ./data/train-mnist.model: output model data
 *           10000  : number of training data
 *           784    : dimension of feature space
 *           1    : C
 *           0.01  : gamma for gaussian kernel
 *           0.001: eps
 *
 * Notes:
 *    		None
 *
 * Output: Lagrangian parameter alphas + support vector = model data 
 *
 * Author: Wei-Hsiang Teng
 * History: 2017/5/25       created
 *          
 */  
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> /* for estimate elapsed time */
#include <math.h>  /* for exp() */
#include <string.h>

#define MAX(x, y) ((x)>(y))?(x):(y)
#define MIN(x, y) ((x)<(y))?(x):(y)
#define ABS(a)      (((a) < 0) ? -(a) : (a))
#define STR_SIZE 1000000

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

/* global variables */
float eps;

/**
* name:        rbf_kernel
*
* description: generate kernel function matrix K(X_i, X_j).
* input:       X: coordinates of training data set	
*              dim: number of dimension of coordinates
*			   size: size of training data set
*              gamma: parameter for guassian kernel: exp(-gamma*|X_i - X_j|^2) 
*
* output:      K: matrix K(X_i, X_j)      
* 
*/
__global__ void rbf_kernel(float* X, float* K, int dim, int size, float gamma)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	
	float ker = 0;
	
	for (int k = 0; k < dim; k++)
	{
		ker += (X[tx * dim + k] - X[ty * dim + k]) * (X[tx * dim + k] - X[ty * dim + k]);
	}
	K[tx * size + ty] = exp(-1 * gamma * ker);
	//if (ty == 0)
	//	printf("(%d %d): %f\n", tx, ty, K[tx * size + ty]);
}

/**
* name:        computeDualityGap
*
* description: computeDualityGap computes parameter DualityGap according to (8).
* input:       Err[]: error function (6)
*              C: regularization
*              b: bias term
*              alphas[]: Lagrangian multipliers
*              ylabel[]: class label for each training data
*              size: size of training data set
*
* output:      DualityGap      
* 
*/
float computeDualityGap(float Err[], float C, float b, float alphas[], int ylabel[], int size)
{
	float DualityGap = 0;
	int i;
	for (i = 0; i < size; i++)
	{
		if (ylabel[i] == 1)
			DualityGap += C*MAX(0, (b - Err[i]));
		else
			DualityGap += C*MAX(0, (-1*b + Err[i]));
		if (alphas[i] != 0)
		{
			DualityGap += alphas[i] * ylabel[i] * Err[i];
		}
	}
	return DualityGap;
}

/**
* name:        computeBupIup
*
* description: computeBupIup computes b_up and I_up according to page 5.
* input:       Err[]: error function (6)
*              C: regularization
*              alphas[]: Lagrangian multipliers
*              ylabel[]: class label for each training data
*              size: size of training data set
*			   b_up: the min error function of sets which unions I0 I1 I2
*              I_up: the index for min of error function of set which unions I0 I1 I2
*	
* output:      None      
* 
*/
void computeBupIup(float Err[], float C, float alphas[], int ylabel[], int size, float *b_up, int *I_up)
{
	int i;
	*b_up = 100000000;
	for (i = 0; i < size; i++)
	{
		if (alphas[i] > 0 && alphas[i] < C)
		{
			if (Err[i] < *b_up) {
				*b_up = Err[i];
				*I_up = i;
				continue;
			}	
		}
		
		if (alphas[i] == 0 && ylabel[i] == 1)
		{
			if (Err[i] < *b_up) {
				*b_up = Err[i];
				*I_up = i;
				continue;
			}	
		}
		
		if (alphas[i] == C && ylabel[i] == -1)
		{
			if (Err[i] < *b_up) {
				*b_up = Err[i];
				*I_up = i;
				continue;
			}	
		}
	}
}

/**
* name:        computeBlowIlow
*
* description: computeBlowIlow computes b_low and I_low according to page 5.
* input:       Err[]: error function (6)
*              C: regularization
*              alphas[]: Lagrangian multipliers
*              ylabel[]: class label for each training data
*              size: size of training data set
*			   b_low: the max error function of sets which unions I0 I3 I4
*              I_low: the index for max of error function of set which unions I0 I3 I4
*	
* output:      None      
* 
*/
void computeBlowIlow(float Err[], float C, float alphas[], int ylabel[], int size, float *b_low, int *I_low)
{
	int i;
	*b_low = -100000000;
	for (i = 0; i < size; i++)
	{
		if (alphas[i] > 0 && alphas[i] < C)
		{
			if (Err[i] > *b_low) {
				*b_low = Err[i];
				*I_low = i;
				continue;
			}	
		}
		
		if (alphas[i] == C && ylabel[i] == 1)
		{
			if (Err[i] > *b_low) {
				*b_low = Err[i];
				*I_low = i;
				continue;
			}		
		}
		
		if (alphas[i] == 0 && ylabel[i] == -1)
		{
			if (Err[i] > *b_low) {
				*b_low = Err[i];
				*I_low = i;
				continue;
			}		
		}
	}
}

/**
* name:        computeNumChaned
*
* description: computeNumChaned implements Procedure takeStep() in page 19.
* input:       I_up: index for minimum of Err in group I0, I1, I2
*			   I_low: index for maximum of Err in group I0, I3, I4
*			   alpha1: alphas[I_up]
*              alpha2: alphas[I_low]
*              X[]: coordinates of training data set
*              y1, y2: Y[I_up], Y[I_low]
*			   F1, F2: Err[I_up], Err[I_low]
*              dim: number of dimension of coordinates
*              Dual: see function (7)
*              C: regularization
*              para_gamma: parameter for guassian kernel
*			   a1, a2: the renewed alpha1, alphas2
*			   K: kernel matrix K(X_i, X_j)
*	
* output:      numChanged      
* 
*/
int computeNumChaned(int I_up, 
                     int I_low, 
					 float alpha1, 
					 float alpha2, 
					 float X[], 
					 int y1, 
					 int y2, 
					 float F1, 
					 float F2, 
					 int dim, 
					 int size,
					 float *Dual, 
					 float C, 
					 float para_gamma, 
					 float* a1, 
					 float* a2,
					 float* K)
{
	if (I_up == I_low) return 0;
	int s = y1 * y2;
	float gamma;
	float L, H, slope, change;
	float k11, k12, k22, eta;
	
	if (y1 == y2)
		gamma = alpha1 + alpha2;
	else
		gamma = alpha1 - alpha2;
	
	if (s == 1)
	{
		L = MAX(0, gamma - C);
		H = MIN(C, gamma);
	} else {
		L = MAX(0, -1*gamma);
		H = MIN(C, C - gamma);
	}
	
	if (H <= L) return 0;
	
	k11 = K[I_up * size + I_up];
	k22 = K[I_low * size + I_low];
	k12 = K[I_up * size + I_low];
	eta = 2*k12 - k11 - k22;
	
	if (eta < eps * (k11 + k22))
	{
		*a2 = alpha2 - (y2*(F1 - F2)/eta);
		if (*a2 < L)
			*a2 = L;
		else if (*a2 > H)
			*a2 = H;
	} else {
		slope = y2*(F1 - F2);
		change = slope * (H - L);
		if (change != 0)
		{
			if (slope > 0)
				*a2 = H;
			else 
				*a2 = L;
		} else {
			*a2 = alpha2;
		}
	}
	
	if (*a2 > C - eps * C) *a2 = C;
	else if (*a2 < eps * C) *a2 = 0;
	
	if (ABS(*a2 - alpha2) < eps * (*a2 + alpha2 + eps)) return 0;
	
	if (s == 1) *a1 = gamma - *a2;
	else *a1 = gamma + *a2;
	
	if (*a1 > C - eps * C) *a1 = C;
	else if (*a1 < eps * C) *a1 = 0;
	
	*Dual = *Dual - (*a1 - alpha1) * (F1 - F2) / y1 + 1 / 2 * eta * (*a1 - alpha2) * (*a1 - alpha2) / y1 / y1;
	return 1;
}

/**
* name:        modified_SMO
*
* description: modified_SMO implements Pseudo-code for the serial SMO in page 19.
* input:       X[]: coordinates of training data set
*              Y[]: class label of [-1 1] for each training data 
*              size: size of training data set
*              dim: number of dimension of coordinates
*              C: regularization
*			   gamma: parameter for guassian kernel
*              tau: condition for convergence
*			   K: kernel matrix K(X_i, X_j)
*	
* output:      alphas[]     
* 
*/
float* modified_SMO(float X[], int Y[], int size, int dim, float C, float gamma, float tau, float *K)
{
	int i;
	float b = 0.0;
	float* alphas;
	float Err[size];
	float b_up, b_low, alpha1, alpha2, a1 = 0, a2 = 0, F1 = 0, F2 = 0;
	int I_up, I_low, y1 = 0, y2 = 0;
	int numChanged;
	float Dual = 0, DualityGap;
	float a1_old, a2_old;
	int num_iter = 0;

	alphas = (float *)malloc(size*sizeof(float));
	
	/* initialize alpha, Err, Dual */
	for (i = 0; i < size; i++) {
		alphas[i] = 0.0;
		Err[i] = -1*Y[i];
	}
	/* initialize b_up, I_up, b_low, I_low, DualityGap */
	DualityGap = computeDualityGap(Err, C, b, alphas, Y, size);
	computeBupIup(Err, C, alphas, Y, size, &b_up, &I_up);
	computeBlowIlow(Err, C, alphas, Y, size, &b_low, &I_low);
	
	numChanged = 1;

	while(DualityGap > tau*ABS(Dual) && numChanged != 0)
	{
		alpha1 = alphas[I_up];
		alpha2 = alphas[I_low];		
		y1 = Y[I_up];
		y2 = Y[I_low];
		F1 = Err[I_up];
		F2 = Err[I_low];
		
		numChanged = computeNumChaned(I_up, I_low, alpha1, alpha2, X, y1, y2, F1, F2, dim, size, &Dual, C, gamma, &a1, &a2, K);
		
		a1_old = alphas[I_up];
		a2_old = alphas[I_low];

		alphas[I_up] = a1;
		alphas[I_low] = a2;
		
		/* update Err[i] */
		for (i = 0; i < size; i++) {
			Err[i] += (alphas[I_up] - a1_old) * Y[I_up] * K[I_up*size+i] 
				+ (alphas[I_low] - a2_old) * Y[I_low] * K[I_low*size+i];  
		}
		
		computeBupIup(Err, C, alphas, Y, size, &b_up, &I_up);
		computeBlowIlow(Err, C, alphas, Y, size, &b_low, &I_low);
		b = (b_low + b_up) / 2;
		DualityGap = computeDualityGap(Err, C, b, alphas, Y, size);
		num_iter++;
		//printf("itertion: %d\n", num_iter);
	}
	
	b = (b_low + b_up) / 2;
	DualityGap = computeDualityGap(Err, C, b, alphas, Y, size);
	
	return alphas;
}

void read_data(char* file, float x[], int y[], int size, int dim)
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
				sscanf(token, "%f %d", &x[i * dim + index - 1], &pre_index);
			index = pre_index;
		    token = strtok(NULL, delim);			
			cnt++;
	    }
	}
	fclose(pFile);
}

void save_model(char* filename, float alphas[], float x[], int y[], float gamma, int size, int dim)
{
	FILE *pFile1; 
	int i, j;
	int total_sv = 0;
	
	pFile1 = fopen(filename, "w"); 
	if (pFile1 == NULL) {
		printf("can't open %s\n", filename);
		exit(-1);
	}
	for (i = 0; i < size; i++) {
		if (alphas[i] != 0)
			total_sv++;
	}
	fprintf(pFile1, "%d %f\n", total_sv, gamma);
	
	for (i = 0; i < size; i++) {
		if (alphas[i] != 0)
		{   
			fprintf(pFile1, "%f", alphas[i]*y[i]);
			for (j = 0; j < dim; j++)
			{
				if (x[i * dim + j] != 0)
					fprintf(pFile1, " %d:%f", j + 1, x[i * dim + j]);
			}
			fprintf(pFile1, "\n");
		}	
	}
	printf("total sv: %d\n", total_sv);
	
	fclose(pFile1);
}

int main(int argc, char* argv[])
{
	int size, dim;
	float* x;
	int *y;
	float C;
	float gamma;
	float tau;
	float* alphas;
	float *kernel;
	float *xd, *kd;
	if (argc < 8) {
		printf("%s data_file model_file data_size data_dim C gamma eps\n", argv[0]);
		exit(-1);
	}
	
	size = atoi(argv[3]);
	dim = atoi(argv[4]);
	C = atof(argv[5]);
	gamma = atof(argv[6]);
	eps = atof(argv[7]);
	
	x = (float *)malloc(size*dim*sizeof(float));
	memset(x, 0, sizeof(float)*size*dim);
	y = (int *)malloc(size*sizeof(int));
	alphas = (float *)malloc(size*sizeof(float));
	kernel = (float *)malloc(size*size*sizeof(float));
	
	read_data(argv[1], x, y, size, dim);
	
	CHECK(cudaMalloc((void **)&xd, size*dim*sizeof(float)));
	CHECK(cudaMalloc((void **)&kd, size*size*sizeof(float)));
	CHECK(cudaMemcpy(xd, x, size*dim*sizeof(float), cudaMemcpyHostToDevice));

	/* start the SMO algorithm */
	tau = 0.000001;

	dim3 block(16, 16);
	dim3 grid((size + block.x - 1)/block.x, (size + block.y - 1)/block.y);
	CHECK(cudaDeviceSynchronize());
	
	rbf_kernel<<<grid, block>>>(xd, kd, dim, size, gamma);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(kernel, kd, size*size*sizeof(float), cudaMemcpyDeviceToHost));
	alphas = modified_SMO(x, y, size, dim, C, gamma, tau, kernel);
	//for (i = 0; i < size; i++) {
	//	printf("(%d %d): %f\n", i, 0, kernel[i*size+0]);
	//}
	
	/* save the result */
	save_model(argv[2], alphas, x, y, gamma, size, dim);

	free(x);
	free(y);
	free(alphas);
	free(kernel);
	cudaFree(xd);
	cudaFree(kd);
	
	// reset device
    CHECK(cudaDeviceReset());
	
	return 0;
}
