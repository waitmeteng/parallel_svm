/** File:     modified_SMO.c
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *
 * Compile:  gcc -Wall -o modified_SMO modified_SMO.c -lm
 * Run:      ex: ./modified_SMO test1_X.txt test1_Y.txt 863 2 2 1 0.1 0.001
 *           test1_X.txt: the coordinates of training data set
 *           test1_Y.txt: the labels of training data set
 *           863  : number of training data
 *           2    : dimension of feature space
 *           2    : number of label class
 *           1    : C
 *           0.1  : sigma for gaussian kernel
 *           0.001: eps
 *
 * Notes:
 *    1.  Follow the paper "Parallel Sequential Minimal Optimization for the Training of Support Vector Machines" by L.J. Cao et al. 2006
 *    2.  Use one-against-all method to implement multiclass version
 *    3.  Modify eps to balance the accuracy and speed (recommend value: 0.001)
 *
 * Output: Lagrangian parameter alphas
 *
 * Author: Wei-Hsiang Teng
 * History: 2017/4/17       created
 *          2017/4/27       modified for multiclass using one-against-all method
 *	        2017/5/4	    add eps to decrease the accuracy but accelerate computing 
 */  
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> // for estimate elapsed time
#include <math.h>  /* for exp() */

#define MAX(x, y) ((x)>(y))?(x):(y)
#define MIN(x, y) ((x)<(y))?(x):(y)
#define ABS(a)      (((a) < 0) ? -(a) : (a))
#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

/* global variables */
double eps;

/**
* name:        kernel
*
* description: kernel generates kernel function K(X_i, X_j) of X which could be linear or gaussian.
* input:       X[]: coordinates of training data set
*              dim: number of dimension of coordinates
*              i, j: index of kernel function K(X_i, X_j)
*              c: 'l' for linear, 'g' for gaussian
*              sigma: deviation for guassian kernel
*
* output:      K(X_i, X_j)      
* 
*/
double kernel(double X[], int dim, int i, int j, char c, double sigma)
{
	double ker = 0.0;
	int m;
	if (c == 'l')
	{
		for (m = 0; m < dim; m++)
		{
			ker += X[i * dim + m] * X[j * dim + m];
		}
	} else if (c == 'g')
	{
		for (m = 0; m < dim; m++)
		{
			ker += (X[i * dim + m] - X[j * dim + m]) * (X[i * dim + m] - X[j * dim + m]);
		}
		ker = exp(-1 * ker / 2 / sigma / sigma);
	} else 
	{
		for (m = 0; m < dim; m++)
		{
			ker += X[i * dim + m] * X[j * dim + m];
		}
	}
	
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
double computeDualityGap(double Err[], double C, double b, double alphas[], double ylabel[], int size)
{
	double DualityGap = 0;
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
void computeBupIup(double Err[], double C, double alphas[], double ylabel[], int size, double *b_up, int *I_up)
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
void computeBlowIlow(double Err[], double C, double alphas[], double ylabel[], int size, double *b_low, int *I_low)
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
* input:       alpha1: alphas[I_up]
*              alpha2: alphas[I_low]
*              X[]: coordinates of training data set
*              ylabel[]: class label for each training data
*			   Err[]: error function (6)
*              dim: number of dimension of coordinates
*              Dual: see function (7)
*              C: regularization
*              sigma: deviation for guassian kernel
*	
* output:      numChanged      
* 
*/
int computeNumChaned(int I_up, int I_low, double alpha1, double alpha2, double X[], int y1, int y2, double F1, double F2, int dim, double *Dual, double C, double sigma, double* a1, double* a2)
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
		L = MAX(0, gamma - C);
		H = MIN(C, gamma);
	} else {
		L = MAX(0, -1*gamma);
		H = MIN(C, C - gamma);
	}
	
	if (H <= L) return 0;
	
	k11 = kernel(X, dim, I_up, I_up, 'g', sigma);
	k22 = kernel(X, dim, I_low, I_low, 'g', sigma);
	k12 = kernel(X, dim, I_up, I_low, 'g', sigma);
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
*              Y[]: class label of [0 1] for each training data 
*              size: size of training data set
*              dim: number of dimension of coordinates
*              C: regularization
*			   sigma: deviation for guassian kernel
*              tau: condition for convergence
*	
* output:      alphas[]     
* 
*/
double* modified_SMO(double X[], int Y[], int size, int dim, double C, double sigma, double tau)
{
	int i;
	double ylabel[size];
	double b = 0.0;
	double* alphas;
	double Err[size];
	double b_up, b_low, alpha1, alpha2, a1 = 0, a2 = 0, F1 = 0, F2 = 0;
	int I_up, I_low, y1 = 0, y2 = 0;
	int numChanged;
	double Dual = 0, DualityGap;
	double a1_old, a2_old;
	
	alphas = (double *)malloc((size + 1)*sizeof(double));
	
	/* initialize alpha, Err, Dual */
	for (i = 0; i < size; i++) {
		if (Y[i] == 0) ylabel[i] = -1;
		else ylabel[i] = 1;
		alphas[i] = 0.0;
		Err[i] = -1*ylabel[i];
	}
	/* initialize b_up, I_up, b_low, I_low, DualityGap */
	DualityGap = computeDualityGap(Err, C, b, alphas, ylabel, size);
	computeBupIup(Err, C, alphas, ylabel, size, &b_up, &I_up);
	computeBlowIlow(Err, C, alphas, ylabel, size, &b_low, &I_low);
	
	numChanged = 1;

	while(DualityGap > tau*ABS(Dual) && numChanged != 0)
	{
		alpha1 = alphas[I_up];
		alpha2 = alphas[I_low];		
		y1 = ylabel[I_up];
		y2 = ylabel[I_low];
		F1 = Err[I_up];
		F2 = Err[I_low];
		
		numChanged = computeNumChaned(I_up, I_low, alpha1, alpha2, X, y1, y2, F1, F2, dim, &Dual, C, sigma, &a1, &a2);
		
		a1_old = alphas[I_up];
		a2_old = alphas[I_low];

		alphas[I_up] = a1;
		alphas[I_low] = a2;
		
		/* update Err[i] */
		for (i = 0; i < size; i++) {
			Err[i] += (alphas[I_up] - a1_old) * ylabel[I_up] * kernel(X, dim, I_up, i, 'g', sigma) 
				+ (alphas[I_low] - a2_old) * ylabel[I_low] * kernel(X, dim, I_low, i, 'g', sigma);  
		}
		
		computeBupIup(Err, C, alphas, ylabel, size, &b_up, &I_up);
		computeBlowIlow(Err, C, alphas, ylabel, size, &b_low, &I_low);
		b = (b_low + b_up) / 2;
		DualityGap = computeDualityGap(Err, C, b, alphas, ylabel, size);
	}
	
	b = (b_low + b_up) / 2;
	DualityGap = computeDualityGap(Err, C, b, alphas, ylabel, size);
	alphas[size] = b;
	
	return alphas;
}



int main(int argc, char* argv[])
{
	int size, dim, class;
	int i, k;
	double* x;
	int *y, *temp_y;
	FILE *pFile, *pFile1; 
	double C;
	double sigma;
	double tau;
	double* alphas;
	double b;
	double start, end;
	char* filename;
	//double* p;
	
	if (argc < 8) {
		printf("%s X_file Y_file paras_file Data_size, Data_dim, Data_class C sigma eps\n", argv[0]);
		exit(-1);
	}
	
	filename = argv[3];
	size = atoi(argv[4]);
	dim = atoi(argv[5]);
	class = atoi(argv[6]);
	C = atof(argv[7]);
	sigma = atof(argv[8]);
	eps = atof(argv[9]);
	
	x = (double *)malloc(size*dim*sizeof(double));
	y = (int *)malloc(size*sizeof(double));
	temp_y = (int *)malloc(size*sizeof(double));
	alphas = (double *)malloc((size+1)*sizeof(double));
	
	pFile = fopen(argv[1], "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", argv[1]);
		exit(-1);
	}
	for (i = 0; i < size*dim; i++)
		fscanf(pFile, "%lf", &x[i]);
	
	pFile = fopen(argv[2], "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", argv[2]);
		exit(-1);
	}
	for (i = 0; i < size; i++)
		fscanf(pFile, "%d", &y[i]);
	
	/* for save the parameters */
	pFile1 = fopen(filename, "w"); 
	if (pFile1 == NULL) {
		printf("can't open %s\n", filename);
		exit(-1);
	} 
	/* start the SMO algorithm */
	tau = 0.000001;

	GET_TIME(start);
	/* one-against-all method */
	for (i = 0; i < class; i++)
	{
		for (k = 0; k < size; k++) {
			if (y[k] == i) temp_y[k] = 1;
			else temp_y[k] = 0;
		}
		alphas = modified_SMO(x, temp_y, size, dim, C, sigma, tau);
		b = -1 * alphas[size];
		
		/* save the result */
		for (k = 0; k < size; k++) {
			fprintf(pFile1, "%lf ", alphas[k]);
		}
		fprintf(pFile1, "%lf\n", b);
	}
	GET_TIME(end);
	printf("The elapsed time is %e seconds\n", end - start);

	free(x);
	free(y);
	fclose(pFile);
	fclose(pFile1);
	return 0;
}
