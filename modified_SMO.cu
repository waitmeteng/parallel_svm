/** File:     modified_SMO.c
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *			 the serial version.
 *
 * Compile:  gcc -Wall -o modified_SMO modified_SMO.c -lm
 * Run:      ex: ./modified_SMO ./data/train-mnist ./data/train-mnist.model 10000 784 1 0.01 0.001
 *           ./data/train-mnist: input training data set
 *           ./data/train-mnist.model: output model data
 *           10000  : number of training data
 *           784    : dimension of feature space
 *           1    : C
 *           0.01  : gamma for gaussian kernel
 *           0.001: eps
 *
 * Notes:
 *    1.  Follow the paper "Parallel Sequential Minimal Optimization for the Training of Support Vector Machines" by L.J. Cao et al. 2006
 *    2.  Use one-against-all method to implement multiclass version (removed)
 *    3.  Modify eps to balance the accuracy and speed (recommend value: 0.001)
 *	  4.  Input file is identical with "libsvm" format
 *    5.  Output file includes support vector
 *
 * Output: Lagrangian parameter alphas + support vector = model data 
 *
 * Author: Wei-Hsiang Teng
 * History: 2017/4/17       created
 *          2017/4/27       modified for multiclass using one-against-all method
 *	        2017/5/4	    add eps to decrease the accuracy but accelerate computing 
 *			2017/5/24       modify the input file to fit "libsvm" format and add support vector to output file
 *			2017/5/28       add execution time profiling
 */  
//#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> /* for estimate elapsed time */
//#include <time.h>
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
        exit(1);                                                               \
    }                                                                          \
}

/* global variables */
float eps;

#if 0
/* for windows */
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    //FILETIME    file_time;
    SYSTEMTIME  system_time;
    //ULARGE_INTEGER ularge;

    GetSystemTime(&system_time);
    //SystemTimeToFileTime(&system_time, &file_time);
    //ularge.LowPart = file_time.dwLowDateTime;
    //ularge.HighPart = file_time.dwHighDateTime;

    //tp->tv_sec = (long) ((ularge.QuadPart - epoch) / 10000000L);
	tp->tv_sec = (long) (system_time.wSecond);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);

    return 0;
}
#endif

double seconds(void)
{
    struct timeval tp;
    //struct timezone tzp;
    int i = gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/**
* name:        rbf_kernel
*
* description: kernel generates kernel function K(X_i, X_j) of X which is gaussian.
* input:       X[]: coordinates of training data set
*              dim: number of dimension of coordinates
*              i, j: index of kernel function K(X_i, X_j)
*              gamma: parameter for guassian kernel: exp(-gamma*|X_i - X_j|^2) 
*
* output:      K(X_i, X_j)      
* 
*/
float rbf_kernel(float X[], int dim, int i, int j, float gamma)
{
	float ker = 0.0;
	int m;
	for (m = 0; m < dim; m++)
	{
		ker += (X[i * dim + m] - X[j * dim + m]) * (X[i * dim + m] - X[j * dim + m]);
	}
	ker = exp(-1 * gamma * ker);
	
	return ker;
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
					 float *Dual, 
					 float C, 
					 float para_gamma, 
					 float* a1, 
					 float* a2)
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
	
	k11 = rbf_kernel(X, dim, I_up, I_up, para_gamma);
	k22 = rbf_kernel(X, dim, I_low, I_low, para_gamma);
	k12 = rbf_kernel(X, dim, I_up, I_low, para_gamma);
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

/**********************************************************************
 *     Initialize alphas and Err
 *********************************************************************/
__global__ void Initialization(float* devErr, float* devAlphas, int* devY, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		devErr[i] = -1*devY[i];
		devAlphas[i] = 0.0;
	}
}

/**********************************************************************
 *     Update f_i
 *********************************************************************/
__global__ void update_fi(float *devErr, float *devX, float a1, float a2, float a1_old, float a2_old, int y1, int y2, int I_up, int I_low, float gamma, int dim, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float k1, k2;
	if (i < size) {
		for (int m = 0; m < dim; m++)
		{
			k1 += (devX[I_up * dim + m] - devX[i * dim + m]) * (devX[I_up * dim + m] - devX[i * dim + m]);
			k2 += (devX[I_low * dim + m] - devX[i * dim + m]) * (devX[I_low * dim + m] - devX[i * dim + m]);
		}
		k1 = expf(-1 * gamma * k1);
		k2 = expf(-1 * gamma * k2);
		devErr[i] = devErr[i] + (a1 - a1_old) * y1 * k1 + (a2 - a2_old) * y2 * k2;  
	}
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
*	
* output:      alphas[]     
* 
*/
float* modified_SMO(float X[], int Y[], int size, int dim, float C, float gamma, float tau)
{

	float b = 0.0;
	float* alphas;
	float* Err;
	float* devX;
	int* devY;
	float* devAlphas;
	float* devErr;
	float b_up, b_low, a1 = 0, a2 = 0, F1 = 0, F2 = 0;
	int I_up, I_low, y1 = 0, y2 = 0;
	int numChanged;
	float Dual = 0, DualityGap = 0;
	float a1_old, a2_old;
	int num_iter = 0;
	double s1, s2, s3, s4;
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	
	alphas = (float *)malloc(size * sizeof(float));
	Err = (float *)malloc(size * sizeof(float));
	CHECK(cudaMalloc((void**)&devX, size*dim*sizeof(float)));
	CHECK(cudaMalloc((void**)&devY, size*sizeof(int)));
	CHECK(cudaMalloc((void**)&devErr, size*sizeof(float)));
	CHECK(cudaMalloc((void**)&devAlphas, size*sizeof(float)));
	CHECK(cudaMemcpy(devX, X, size*dim*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(devY, Y, size*sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 block(256);
	dim3 grid((size + block.x - 1)/block.x);
	
	Initialization<<<grid, block>>>(devErr, devAlphas, devY, size);
	
	/* initialize b_up, I_up, b_low, I_low, DualityGap */
	CHECK(cudaMemcpy(Err, devErr, size * sizeof(float), cudaMemcpyDeviceToHost)); 
	CHECK(cudaMemcpy(alphas, devAlphas, size * sizeof(float), cudaMemcpyDeviceToHost)); 
	DualityGap = computeDualityGap(Err, C, b, alphas, Y, size);
	printf("DualityGap: %f\n", DualityGap);
	computeBupIup(Err, C, alphas, Y, size, &b_up, &I_up);
	computeBlowIlow(Err, C, alphas, Y, size, &b_low, &I_low);
	printf("b_up: %f, b_low: %f\n", b_up, b_low);
	numChanged = 1;
	
	while(DualityGap > tau*ABS(Dual) && numChanged != 0)
	{	
		s1 = seconds();
		a1_old = alphas[I_up];
		a2_old = alphas[I_low];	
		y1 = Y[I_up];
		y2 = Y[I_low];
		F1 = Err[I_up];
		F2 = Err[I_low];
		numChanged = computeNumChaned(I_up, I_low, a1_old, a2_old, X, y1, y2, F1, F2, dim, &Dual, C, gamma, &a1, &a2);
		alphas[I_up] = a1;
		alphas[I_low] = a2;
		CHECK(cudaMemcpy(devAlphas, alphas, size * sizeof(float), cudaMemcpyHostToDevice)); 
		t1 += (seconds() - s1);

		/* update Err[i] */
		s2 = seconds();
		update_fi<<<grid, block>>>(devErr, devX, a1, a2, a1_old, a2_old, y1, y2, I_up, I_low, gamma, dim, size);
		CHECK(cudaMemcpy(Err, devErr, size * sizeof(float), cudaMemcpyDeviceToHost)); 
		t2 += (seconds() - s2);
		
		s3 = seconds();
		computeBupIup(Err, C, alphas, Y, size, &b_up, &I_up);
		computeBlowIlow(Err, C, alphas, Y, size, &b_low, &I_low);
		b = (b_low + b_up) / 2;
		t3 += (seconds() - s3);
		
		s4 = seconds();
		DualityGap = computeDualityGap(Err, C, b, alphas, Y, size);
		t4 += (seconds() - s4);
		
		num_iter++;
		//printf("itertion: %d\n", num_iter);
	}	
	//b = (b_low + b_up) / 2;
	//DualityGap = computeDualityGap(Err, C, b, alphas, Y, size);
	printf("computeNumChaned    : %lf secs\n", t1);
	printf("update f_i          : %lf secs\n", t2);
	printf("update b_up, b_low  : %lf secs\n", t3);
	printf("computeDualityGap   : %lf secs\n", t4);
	
	cudaFree(devErr);
	cudaFree(devAlphas);
	cudaFree(devX);
	cudaFree(devY);
	
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
	double start, end;

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
	printf("start reading data\n");
	fflush(stdout);
	read_data(argv[1], x, y, size, dim);
	printf("finish reading data\n");
	fflush(stdout);
	/* start the SMO algorithm */
	tau = 0.000001;

	start = seconds();
	alphas = modified_SMO(x, y, size, dim, C, gamma, tau);
	end = seconds();
	printf("The total elapsed time is %lf seconds\n", end - start);
	
	/* save the result */
	save_model(argv[2], alphas, x, y, gamma, size, dim);

	free(x);
	free(y);
	
	return 0;
}
