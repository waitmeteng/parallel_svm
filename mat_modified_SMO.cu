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
 * Output: Lagrangian parameter alphas + support vector + b = model data 
 *
 * Author: Wei-Hsiang Teng
 * History: 2017/5/25       created
 *			2017/5/27		use the shared memory of GPU to speed up the calculation.
 *          2017/6/3        code refactoring   
 *          2017/6/13       change data type float to double 
 */  
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> /* for estimate elapsed time */
#include <math.h>  /* for exp() */
#include <string.h>
#include <limits.h>

#define MAX(x, y) ((x)>(y))?(x):(y)
#define MIN(x, y) ((x)<(y))?(x):(y)
#define ABS(a)      (((a) < 0) ? -(a) : (a))
#define STR_SIZE 8192
#define TILE_SIZE 16
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

struct problem
{
	double* x;			/* input features */
	double* alphas;		/* output Lagrangian parameters */
	int *y;				/* input labels */
	int size;			/* size of training data set */
	int	dim;			/* number of dimension of coordinates */
	double C;			/* regularization parameter */
	double gamma;		/* parameter for gaussian kernel function */
	double b;			/* offset of decision boundary */
	double tau;			/* parameter for divergence */
	double eps;			/* tolerance */
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


/**********************************************************************
 *     calculate kernel function parallelly
 *********************************************************************/
__global__ void rbf_kernel(double* devX, double* devK, int dim, int size, double gamma)
{
#if 0
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	
	double ker = 0;
	
	for (int k = 0; k < dim; k++)
	{
		ker += (devX[tx * dim + k] - devX[ty * dim + k]) * (devX[tx * dim + k] - devX[ty * dim + k]);
	}
	devK[ty * size + tx] = exp(-1 * gamma * ker);
	//if (ty == 0)
	//	printf("(%d %d): %f\n", tx, ty, devK[tx * size + ty]);
#else
	__shared__ double Xs1[TILE_SIZE][TILE_SIZE];
	__shared__ double Xs2[TILE_SIZE][TILE_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_SIZE + ty;
	int Col = bx * TILE_SIZE + tx;

	double ker = 0;
	for (int m = 0; m < dim / TILE_SIZE; m++) {
		Xs1[ty][tx] = devX[Row * dim + (m * TILE_SIZE + tx)];
		Xs2[ty][tx] = devX[Col * dim + (m * TILE_SIZE + ty)];
		__syncthreads();
		for (int k = 0; k < TILE_SIZE; k++) {
			ker += (Xs1[ty][k] - Xs2[k][tx]) * (Xs1[ty][k] - Xs2[k][tx]);
		}
		__syncthreads();
	}
	devK[Row * size + Col] = exp(-1 * gamma * ker);
#endif
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
double computeDualityGap(double Err[], struct problem* prob)
{
	double DualityGap = 0;
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
void computeBupIup(double Err[], struct problem* prob, double *b_up, int *I_up)
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
void computeBlowIlow(double Err[], struct problem* prob, double *b_low, int *I_low)
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
					 double alpha1, 
					 double alpha2,  
					 int y1, 
					 int y2, 
					 double F1, 
					 double F2, 
					 double *Dual, 
					 double* a1, 
					 double* a2,
					 double* K)
{
	if (I_up == I_low) return 0;
	int s = y1 * y2;
	double gamma;
	double L, H, slope, change;
	double k11, k12, k22, eta;
	
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
	
	k11 = K[I_up * prob->size + I_up];
	k22 = K[I_low * prob->size + I_low];
	k12 = K[I_up * prob->size + I_low];
	eta = 2*k12 - k11 - k22;
	
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

/**
* name:        modified_SMO
*
* description: modified_SMO implements Pseudo-code for the serial SMO in page 19.
* input:       prob: needed information describing the problem
*	
* output:      None     
* 
*/
void modified_SMO(struct problem* prob, double *K)
{
	int i;
	prob->b = 0.0;
	double* Err;
	double b_up, b_low, a1 = 0, a2 = 0, F1 = 0, F2 = 0;
	int I_up, I_low, y1 = 0, y2 = 0;
	int numChanged;
	double Dual = 0, DualityGap;
	double a1_old, a2_old;
	int num_iter = 0;
	
	Err = (double *)malloc(sizeof(double) * prob->size);
	
	/* initialize alpha, Err, Dual */
	for (i = 0; i < prob->size; i++) {
		prob->alphas[i] = 0.0;
		Err[i] = -1 * prob->y[i];
	}
	/* initialize b_up, I_up, b_low, I_low, DualityGap */
	DualityGap = computeDualityGap(Err, prob);
	computeBupIup(Err, prob, &b_up, &I_up);
	computeBlowIlow(Err, prob, &b_low, &I_low);
	
	numChanged = 1;

	while(DualityGap > prob->tau*ABS(Dual) && numChanged != 0)
	{
		a1_old = prob->alphas[I_up];
		a2_old = prob->alphas[I_low];	
		y1 = prob->y[I_up];
		y2 = prob->y[I_low];
		F1 = Err[I_up];
		F2 = Err[I_low];
		
		numChanged = computeNumChaned(prob, I_up, I_low, a1_old, a2_old, y1, y2, F1, F2, &Dual, &a1, &a2, K);

		prob->alphas[I_up] = a1;
		prob->alphas[I_low] = a2;
		
		/* update Err[i] */
		for (i = 0; i < prob->size; i++) {
			Err[i] += (a1 - a1_old) * y1 * K[I_up * prob->size + i] 
				+ (a2 - a2_old) * y2 * K[I_low * prob->size + i];  
		}
		
		computeBupIup(Err, prob, &b_up, &I_up);
		computeBlowIlow(Err, prob, &b_low, &I_low);
		prob->b = (b_low + b_up) / 2;
		DualityGap = computeDualityGap(Err, prob);
		num_iter++;
		//printf("itertion: %d\n", num_iter);
	}
	
	prob->b = -1 * (b_low + b_up) / 2;
	printf("b = %f\n", prob->b);
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
				sscanf(token, "%lf %d", &prob->x[i * prob->dim + index - 1], &pre_index);
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
	fprintf(pFile, "%d %lf %lf\n", total_sv, prob->gamma, prob->b);
	
	for (i = 0; i < prob->size; i++) {
		if (prob->alphas[i] != 0)
		{   
			fprintf(pFile, "%lf", prob->alphas[i] * prob->y[i]);
			for (j = 0; j < prob->dim; j++)
			{
				if (prob->x[i * prob->dim + j] != 0)
					fprintf(pFile, " %d:%lf", j + 1, prob->x[i * prob->dim + j]);
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
	double *kernel;
	double start, elapsed;
	
	/* device variables */
	double *xd, *kd;
	
	if (argc < 8) {
		printf("%s data_file model_file data_size data_dim C gamma eps\n", argv[0]);
		exit(-1);
	}
	
	start = seconds();
	prob->size = atoi(argv[3]);
	prob->dim = atoi(argv[4]);
	prob->C = atof(argv[5]);
	prob->gamma = atof(argv[6]);
	prob->eps = atof(argv[7]);
	
	prob->x = (double *)malloc(prob->size * prob->dim * sizeof(double));
	memset(prob->x, 0, sizeof(double) * prob->size * prob->dim);
	prob->y = (int *)malloc(prob->size * sizeof(int));
	prob->alphas = (double *)malloc(prob->size * sizeof(double));
	kernel = (double *)malloc(prob->size * prob->size * sizeof(double));
	if (kernel == NULL)
	{
		printf("malloc fails at %s, %d\n", __func__, __LINE__);
		exit(-1);
	}
	
	CHECK(cudaMalloc((void **)&xd, prob->size * prob->dim * sizeof(double)));
	CHECK(cudaMalloc((void **)&kd, prob->size * prob->size * sizeof(double)));
	
	read_data(argv[1], prob);
	
	CHECK(cudaMemcpy(xd, prob->x, prob->size * prob->dim * sizeof(double), cudaMemcpyHostToDevice));
	
	/* start the SMO algorithm */
	prob->tau = 0.000001;

	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid((prob->size + block.x - 1)/block.x, (prob->size + block.y - 1)/block.y);
	CHECK(cudaDeviceSynchronize());
	
	rbf_kernel<<<grid, block>>>(xd, kd, prob->dim, prob->size, prob->gamma);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(kernel, kd, prob->size * prob->size * sizeof(double), cudaMemcpyDeviceToHost));
	modified_SMO(prob, kernel);
	//for (i = 0; i < size; i++) {
	//	printf("(%d %d): %f\n", i, 0, kernel[i*size+0]);
	//}
	/* save the result */
	save_model(argv[2], prob);

	free(prob->x);
	free(prob->y);
	free(prob->alphas);
	free(kernel);
	cudaFree(xd);
	cudaFree(kd);
	
	// reset device
    CHECK(cudaDeviceReset());
	
	elapsed = seconds() - start;
	printf("The total elapsed time is %lf seconds\n", elapsed);
	
	return 0;
}
