/** File:     modified_SMO.cu
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *			 the CUDA version.
 *
 * Compile:  nvcc -o modified_SMO modified_SMO.cu
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
 *    6.  The number of training data should be the same with the one in the input file or the program will crash
 *
 * Output: Lagrangian parameter alphas + support vector + b = model data 
 *
 * Author: Wei-Hsiang Teng
 * History:   2017/6/2        created
 *
 */  

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> /* for estimate elapsed time */
#include <math.h>  /* for exp() */
#include <string.h>
#include <limits.h>

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
#define MAX(x, y) ((x)>(y))?(x):(y)
#define MIN(x, y) ((x)<(y))?(x):(y)
#define ABS(a)      (((a) < 0) ? -(a) : (a))
#define STR_SIZE 8192

struct problem
{
	float* x;			/* input features */
	float* alphas;		/* output Lagrangian parameters */
	int *y;				/* input labels */
	int size;			/* size of training data set */
	int	dim;			/* number of dimension of coordinates */
	float C;			/* regularization parameter */
	float gamma;		/* parameter for gaussian kernel function */
	float b;			/* offset of decision boundary */
	float tau;			/* parameter for divergence */
	float eps;			/* tolerance */
};
/**
* name:        seconds
*
* description: for estimating execution time   
* 
*/
double seconds(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/**
* name:        rbf_kernel
*
* description: generate kernel function K(X_i, X_j) of X which is gaussian.
* input:       prob: needed information describing the problem
*              i, j: index of kernel function K(X_i, X_j)
*
* output:      K(X_i, X_j)      
* 
*/
float rbf_kernel(struct problem* prob, int i, int j)
{
	float ker = 0.0;
	int m;
	
	for (m = 0; m < prob->dim; m++)
	{
		ker += (prob->x[i * prob->dim + m] - prob->x[j * prob->dim + m]) * (prob->x[i * prob->dim + m] - prob->x[j * prob->dim + m]);
	}
	ker = exp(-1 * prob->gamma * ker);
	
	return ker;
}

/**
* name:        computeDualityGap
*
* description: computeDualityGap computes parameter DualityGap according to (8).
* input:       Err[]: error function (6)
*              prob: needed information describing the problem
*
* output:      DualityGap      
* 
*/
float computeDualityGap(float Err[], struct problem* prob)
{
	float DualityGap = 0;
	int i;
	
	for (i = 0; i < prob->size; i++)
	{
		if (prob->y[i] == 1)
			DualityGap += prob->C * MAX(0, (prob->b - Err[i]));
		else
			DualityGap += prob->C * MAX(0, (-1 * prob->b + Err[i]));
		if (prob->alphas[i] != 0)
		{
			DualityGap += prob->alphas[i] * prob->y[i] * Err[i];
		}
	}
	return DualityGap;
}

/**
* name:        computeBupIup
*
* description: computeBupIup computes b_up and I_up according to page 5.
* input:       Err[]: error function (6)
*              prob: needed information describing the problem
*			   b_up: the min error function of sets which unions I0 I1 I2
*              I_up: the index for min of error function of set which unions I0 I1 I2
*	
* output:      None      
* 
*/
void computeBupIup(float Err[], struct problem* prob, float *b_up, int *I_up)
{
	int i;
	*b_up = INT_MAX;
	
	for (i = 0; i < prob->size; i++)
	{
		if (prob->alphas[i] > 0 && prob->alphas[i] < prob->C)
		{
			if (Err[i] < *b_up) {
				*b_up = Err[i];
				*I_up = i;
				continue;
			}	
		}
		
		if (prob->alphas[i] == 0 && prob->y[i] == 1)
		{
			if (Err[i] < *b_up) {
				*b_up = Err[i];
				*I_up = i;
				continue;
			}	
		}
		
		if (prob->alphas[i] == prob->C && prob->y[i] == -1)
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
*              prob: needed information describing the problem
*			   b_low: the max error function of sets which unions I0 I3 I4
*              I_low: the index for max of error function of set which unions I0 I3 I4
*	
* output:      None      
* 
*/
void computeBlowIlow(float Err[], struct problem* prob, float *b_low, int *I_low)
{
	int i;
	*b_low = INT_MIN;
	
	for (i = 0; i < prob->size; i++)
	{
		if (prob->alphas[i] > 0 && prob->alphas[i] < prob->C)
		{
			if (Err[i] > *b_low) {
				*b_low = Err[i];
				*I_low = i;
				continue;
			}	
		}
		
		if (prob->alphas[i] == prob->C && prob->y[i] == 1)
		{
			if (Err[i] > *b_low) {
				*b_low = Err[i];
				*I_low = i;
				continue;
			}		
		}
		
		if (prob->alphas[i] == 0 && prob->y[i] == -1)
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
* input:       prob: needed information describing the problem
*			   I_up: index for minimum of Err in group I0, I1, I2
*			   I_low: index for maximum of Err in group I0, I3, I4
*			   alpha1: alphas[I_up]
*              alpha2: alphas[I_low]
*              y1, y2: Y[I_up], Y[I_low]
*			   F1, F2: Err[I_up], Err[I_low]
*              Dual: see function (7)
*			   a1, a2: the renewed alpha1, alphas2
*	
* output:      numChanged      
* 
*/
int computeNumChaned(struct problem* prob,
					 int I_up, 
                     int I_low, 
					 float alpha1, 
					 float alpha2,  
					 int y1, 
					 int y2, 
					 float F1, 
					 float F2, 
					 float *Dual, 
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
		L = MAX(0, gamma - prob->C);
		H = MIN(prob->C, gamma);
	} else {
		L = MAX(0, -1 * gamma);
		H = MIN(prob->C, prob->C - gamma);
	}
	
	if (H <= L) return 0;
	
	k11 = rbf_kernel(prob, I_up, I_up);
	k22 = rbf_kernel(prob, I_low, I_low);
	k12 = rbf_kernel(prob, I_up, I_low);
	eta = 2 * k12 - k11 - k22;
	
	if (eta < prob->eps * (k11 + k22))
	{
		*a2 = alpha2 - (y2 * (F1 - F2) / eta);
		if (*a2 < L)
			*a2 = L;
		else if (*a2 > H)
			*a2 = H;
	} else {
		slope = y2 * (F1 - F2);
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
	
	if (*a2 > prob->C - prob->eps * prob->C) *a2 = prob->C;
	else if (*a2 < prob->eps * prob->C) *a2 = 0;
	
	if (ABS(*a2 - alpha2) < prob->eps * (*a2 + alpha2 + prob->eps)) return 0;
	
	if (s == 1) *a1 = gamma - *a2;
	else *a1 = gamma + *a2;
	
	if (*a1 > prob->C - prob->eps * prob->C) *a1 = prob->C;
	else if (*a1 < prob->eps * prob->C) *a1 = 0;
	
	*Dual = *Dual - (*a1 - alpha1) * (F1 - F2) / y1 + 1 / 2 * eta * (*a1 - alpha2) * (*a1 - alpha2) / y1 / y1;
	return 1;
}

/**********************************************************************
 *     Initialize alphas and Err
 *********************************************************************/
__global__ void Initialization(float* devErr, int* devY, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		devErr[i] = -1*devY[i];
	}
}

/**********************************************************************
 *     Update f_i
 *********************************************************************/
__global__ void update_fi(float *devErr, float *devX, float a1, float a2, float a1_old, float a2_old, int y1, int y2, int I_up, int I_low, float gamma, int dim, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float k1 = 0, k2 = 0;
	if (i < size) {
		for (int m = 0; m < dim; m++)
		{
			k1 += (devX[I_up * dim + m] - devX[i * dim + m]) * (devX[I_up * dim + m] - devX[i * dim + m]);
			k2 += (devX[I_low * dim + m] - devX[i * dim + m]) * (devX[I_low * dim + m] - devX[i * dim + m]);
		}
		k1 = expf(-1 * gamma * k1);
		k2 = expf(-1 * gamma * k2);
		devErr[i] += (a1 - a1_old) * y1 * k1 + (a2 - a2_old) * y2 * k2;  
	}
}

/**
* name:        modified_SMO
*
* description: modified_SMO implements Pseudo-code for the serial SMO in page 19.
* input:       prob: needed information describing the problem
*	
* output:      None     
* 
*/
void modified_SMO(struct problem* prob)
{
	prob->b = 0.0;
	float* Err;
	float b_up, b_low, a1 = 0, a2 = 0, F1 = 0, F2 = 0;
	int I_up, I_low, y1 = 0, y2 = 0;
	int numChanged;
	float Dual = 0, DualityGap;
	float a1_old, a2_old;
	int num_iter = 0;
	double s1, s2, s3, s4;
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0;

	/* device variables */
	float* devX;
	int* devY;
	float* devErr;
	
	Err = (float *)malloc(sizeof(float) * prob->size);
	
	/* allocate device memory */
	CHECK(cudaMalloc((void**)&devX, prob->size * prob->dim * sizeof(float)));
	CHECK(cudaMalloc((void**)&devY, prob->size * sizeof(int)));
	CHECK(cudaMalloc((void**)&devErr, prob->size * sizeof(float)));
	CHECK(cudaMemcpy(devX, prob->x, prob->size * prob->dim * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(devY, prob->y, prob->size * sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 block(32);
	dim3 grid((prob->size + block.x - 1)/block.x);
	
	Initialization<<<grid, block>>>(devErr, devY, prob->size);
	memset(prob->alphas, 0, sizeof(float) * prob->size);
	
	/* initialize b_up, I_up, b_low, I_low, DualityGap */
	CHECK(cudaMemcpy(Err, devErr, prob->size * sizeof(float), cudaMemcpyDeviceToHost));  
	DualityGap = computeDualityGap(Err, prob);
	computeBupIup(Err, prob, &b_up, &I_up);
	computeBlowIlow(Err, prob, &b_low, &I_low);
	numChanged = 1;
	
	while(DualityGap > prob->tau*ABS(Dual) && numChanged != 0)
	{	
		s1 = seconds();
		a1_old = prob->alphas[I_up];
		a2_old = prob->alphas[I_low];	
		y1 = prob->y[I_up];
		y2 = prob->y[I_low];
		F1 = Err[I_up];
		F2 = Err[I_low];
		numChanged = computeNumChaned(prob, I_up, I_low, a1_old, a2_old, y1, y2, F1, F2, &Dual, &a1, &a2);
		prob->alphas[I_up] = a1;
		prob->alphas[I_low] = a2; 
		t1 += (seconds() - s1);

		/* update Err[i] */
		s2 = seconds();
		update_fi<<<grid, block>>>(devErr, devX, a1, a2, a1_old, a2_old, y1, y2, I_up, I_low, prob->gamma, prob->dim, prob->size);
		CHECK(cudaMemcpy(Err, devErr, prob->size * sizeof(float), cudaMemcpyDeviceToHost)); 
		t2 += (seconds() - s2);
		
		s3 = seconds();
		computeBupIup(Err, prob, &b_up, &I_up);
		computeBlowIlow(Err, prob, &b_low, &I_low);
		prob->b = (b_low + b_up) / 2;
		t3 += (seconds() - s3);
		
		s4 = seconds();
		DualityGap = computeDualityGap(Err, prob);
		t4 += (seconds() - s4);
		
		num_iter++;
		//printf("itertion: %d\n", num_iter);
	}	
	prob->b = -1 * (b_low + b_up) / 2;

	printf("computeNumChaned    : %lf secs\n", t1);
	printf("update f_i          : %lf secs\n", t2);
	printf("update b_up, b_low  : %lf secs\n", t3);
	printf("computeDualityGap   : %lf secs\n", t4);
	printf("b = %f\n", prob->b);
	
	cudaFree(devErr);
	cudaFree(devX);
	cudaFree(devY);

}

void read_data(char* file, struct problem* prob)
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
	
	for (i = 0; i < prob->size; i++)
	{
		int cnt = 0;
		fgets(s, sizeof(s), pFile);
	    /* get the first token */
	    token = strtok(s, delim);
	    sscanf(token, "%d %d", &prob->y[i], &index);
	    /* walk through other tokens */
	    while( token != NULL ) 
	    {
			if (cnt == 0) {
				token = strtok(NULL, delim);
			}
			if (index >= 1 && index <= prob->dim)
				sscanf(token, "%f %d", &prob->x[i * prob->dim + index - 1], &pre_index);
			index = pre_index;
		    token = strtok(NULL, delim);			
			cnt++;
	    }
		
	}
	fclose(pFile);
}

void save_model(char* filename, struct problem* prob)
{
	FILE *pFile; 
	int i, j;
	int total_sv = 0;
	
	pFile = fopen(filename, "w"); 
	if (pFile == NULL) {
		printf("can't open %s\n", filename);
		exit(-1);
	}
	for (i = 0; i < prob->size; i++) {
		if (prob->alphas[i] != 0)
			total_sv++;
	}
	fprintf(pFile, "%d %f %f\n", total_sv, prob->gamma, prob->b);
	
	for (i = 0; i < prob->size; i++) {
		if (prob->alphas[i] != 0)
		{   
			fprintf(pFile, "%f", prob->alphas[i] * prob->y[i]);
			for (j = 0; j < prob->dim; j++)
			{
				if (prob->x[i * prob->dim + j] != 0)
					fprintf(pFile, " %d:%f", j + 1, prob->x[i * prob->dim + j]);
			}
			fprintf(pFile, "\n");
		}	
	}
	printf("total sv: %d\n", total_sv);
	
	fclose(pFile);
}

int main(int argc, char* argv[])
{
	struct problem* prob = (struct problem*)malloc(sizeof(*prob));
	double start, end;

	if (argc < 8) {
		printf("%s data_file model_file data_size data_dim C gamma eps\n", argv[0]);
		exit(-1);
	}

	prob->size = atoi(argv[3]);
	prob->dim = atoi(argv[4]);
	prob->C = atof(argv[5]);
	prob->gamma = atof(argv[6]);
	prob->eps = atof(argv[7]);

	prob->x = (float *)malloc(prob->size * prob->dim * sizeof(float));
	memset(prob->x, 0, sizeof(float) * prob->size * prob->dim);
	prob->y = (int *)malloc(prob->size * sizeof(int));
	prob->alphas = (float *)malloc(prob->size * sizeof(float));
	
	read_data(argv[1], prob);

	/* start the SMO algorithm */
	prob->tau = 0.000001;

	start = seconds();
	modified_SMO(prob);
	end = seconds();
	printf("The total elapsed time is %lf seconds\n", end - start);
	
	/* save the result */
	save_model(argv[2], prob);

	free(prob->x);
	free(prob->y);
	free(prob->alphas);
	
	return 0;
}
