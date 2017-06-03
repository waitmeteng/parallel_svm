/** File:    svmPredict.c
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *           use to get accuracy of training result
 *
 * Compile:  gcc -Wall -o svmPredict svmPredict.c -lm
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
 * History: 2017/4/27       created
 *	        2017/5/24	    fix the rbf_kernel bug: in K(X_i, X_j), we have to use support vector for X_i, and input test data for X_j
 */  

#include <string.h> /* for memset */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> // for estimate elapsed time
#include <math.h>
#define STR_SIZE 8192
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
float rbf_kernel(float x1[], float x2[], int i, int j, int dim, float gamma)
{
	float ker = 0.0;
	int m;

	for (m = 0; m < dim; m++)
	{
		ker += (x1[i * dim + m] - x2[j * dim + m]) * (x1[i * dim + m] - x2[j * dim + m]);
	}
	ker = exp(-1 * gamma * ker);
	
	return ker;
}

float svmPredict(float x1[], float x2[], int y1[], float alphas[], int size, int total_sv, int dim, float gamma, float b)
{
	int i, j, result;
	float dual;
	int num = 0;
	int iter = 0;
	for (i = 0; i < size; i++) {
		dual = 0;
		for (j = 0; j < total_sv; j++) {
			dual +=  alphas[j] * rbf_kernel(x1, x2, i, j, dim, gamma);
		} 
		dual += b;
		result = 1;
		if (dual < 0)
			result = -1;
		if (result == y1[i])
			num++;
		printf("iteration: %d\n", iter++);
	}
	
	return ((float)num/size);
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

void read_model(char* file, float x[], float alphas[], int dim, int total_sv)
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
	        sscanf(token, "%f %d", &alphas[i], &index);
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

int main(int argc, char* argv[])
{
	int size, dim, total_sv;
	float* x1, *x2;
	float gamma, b;
	int *y1;
	float* alphas;
	float accuracy;
	double start, end;
	
	if (argc < 5) {
		printf("%s data_file model_file data_size data_dim\n", argv[0]);
		exit(-1);
	}
	
	size = atoi(argv[3]);
	dim = atoi(argv[4]);
	
	x1 = (float *)malloc(size*dim*sizeof(float));
	memset(x1, 0, sizeof(float)*size*dim);
	y1 = (int *)malloc(size*sizeof(float));
	
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
	fscanf(fp, "%d %f %f", &total_sv, &gamma, &b);
	fclose(fp);
	x2 = (float *)malloc(total_sv*dim*sizeof(float));
	memset(x2, 0, sizeof(float)*total_sv*dim);
	alphas = (float *)malloc(total_sv*sizeof(float));
	read_model(argv[2], x2, alphas, dim, total_sv);
	
	GET_TIME(start);
	accuracy = svmPredict(x1, x2, y1, alphas, size, total_sv, dim, gamma, b);
	GET_TIME(end);
	printf("accuracy: %1.5f\n", accuracy);
	printf("elapsed time is %lf seconds\n", end - start);
	free(y1);
    	free(x2);
	free(alphas);
	return 0;
}
