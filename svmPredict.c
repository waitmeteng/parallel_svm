/** File:    svmPredict.c
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *
 * Compile:  gcc -Wall -o svmPredict svmPredict.c -lm
 * Run:      ./svmPredict paras.dat test2_X.txt test2_Y.txt 5000 400 10 0.1
 *           test2_X.txt: the coordinates of training data set
 *           test2_Y.txt: the labels of training data set
 *           5000       : number of training data 
 *           400        : dimension of feature space
 *           10         : number of label class
 *           0.1        : sigma for guassian kernel
 *
 * Notes:
 *    1.  Need modified_SMO executable to train alphas to predict the accuracy of input data set. 
 *
 * Output: percentage of prediction accuracy of input data
 *
 * Author: Wei-Hsiang Teng
 * History: 2017/4/27       created
 *
 */  

#include <string.h> /* for memset */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> // for estimate elapsed time
#include <math.h>

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}
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
	if (c == 'g')
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

int argMax(double p[], int class)
{
	int i, pred;
	double max_val = -1000000;
	for (i = 0; i < class; i++) {
		if (p[i] > max_val) {
			max_val = p[i];
			pred = i;
		}
	}
	return pred;
}

int *svmPredict(double X[], int y[], double alphas[], int size, int dim, int class, char c, double sigma)
{
	int i, j, k, ylabel;
	int * pred = (int *)malloc(size * sizeof(double));
	double* prediction = (double *)malloc(class * sizeof(double));
	
	
	for (i = 0; i < size; i++) {
		memset(prediction, 0, class * sizeof(double));
		for (k = 0; k < class; k++) {
			for (j = 0; j < size; j++) {
				if (y[j] == k) ylabel = 1;
				else ylabel = -1;
				prediction[k] +=  alphas[k * (size + 1) + j] * ylabel * kernel(X, dim, j, i, 'g', sigma);
			} 
			prediction[k] += alphas[size + k * (size + 1)];
		}
		
		pred[i] = argMax(prediction, class);
	}
	
	return pred;
}

int main(int argc, char* argv[])
{
	int size, dim, class;
	int i;
	double* x;
	double sigma;
	int *y;
	double* alphas;
	FILE *pFile; 
	int* p;
	double accuracy = 0;
	double start, end;
	
	if (argc < 8) {
		printf("%s alphas_file X_file Y_file Data_size, Data_dim, Data_class sigma\n", argv[0]);
		exit(-1);
	}
	
	size = atoi(argv[4]);
	dim = atoi(argv[5]);
	class = atoi(argv[6]);
	sigma = atof(argv[7]);
	
	x = (double *)malloc(size*dim*sizeof(double));
	y = (int *)malloc(size*sizeof(double));
	alphas = (double *)malloc((size + 1) * class * sizeof(double));
	p = (int *)malloc(size*sizeof(double));
	
	/* read files */
	pFile = fopen(argv[1], "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", argv[1]);
		exit(-1);
	}
	for (i = 0; i < (size + 1)*class; i++)
		fscanf(pFile, "%lf", &alphas[i]);
	
	pFile = fopen(argv[2], "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", argv[2]);
		exit(-1);
	}
	for (i = 0; i < size*dim; i++)
		fscanf(pFile, "%lf", &x[i]);
	
	pFile = fopen(argv[3], "r"); 
	if (pFile == NULL) {
		printf("can't open %s\n", argv[3]);
		exit(-1);
	}
	for (i = 0; i < size; i++)
		fscanf(pFile, "%d", &y[i]);

	GET_TIME(start);
	p = svmPredict(x, y, alphas, size, dim, class, 'g', sigma);
	GET_TIME(end);
	for (i = 0; i < size; i++) {
		//printf("%d\n", p[i]);
		if (p[i] == y[i]) accuracy++;
	}
	
	accuracy /= size;
	printf("accuracy: %lf\n", accuracy);
	printf("The elapsed time is %e seconds\n", end - start);
	free(alphas);
	free(p);
	free(x);
	free(y);
	fclose(pFile);

	return 0;
}
