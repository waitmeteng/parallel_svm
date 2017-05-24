/** File:     modified_SMO.c
 * Purpose:  Parallel Programming 2017 Final Project: Training Support Vector Machine on multiprocessors and GPUs
 *
 * Compile:  gcc -Wall -o mpi_modified_SMO mpi_modified_SMO.c -lm
 * Run:      ./mpi_modified_SMO test1_X.txt test1_Y.txt 863 "filename" 2 2 1 0.1 0.001
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
#include <mpi.h>

#define MAX(x, y) ((x)>(y))?(x):(y)
#define MIN(x, y) ((x)<(y))?(x):(y)
#define ABS(a)      (((a) < 0) ? -(a) : (a))
#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}
#define PRINT(fmt, args...) printf("%d, " fmt, my_rank, ## args)


typedef struct smo {
	int I_up, I_low, z1, z2, y1, y2;
	double b_up, b_low, alpha1, alpha2, F1, F2, a1, a2, a1_old, a2_old, Dual;
} SMO;


enum {
	I_UP,
	I_LOW
};

/* global variables */
double eps;
int my_rank, comm_sz;
MPI_Datatype smoType;
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
int computeNumChaned(double X[], SMO *smo, int dim, double C, double sigma)
{
	if (smo->I_up == smo->I_low) {
		return 0;
	}
	int s = smo->y1 * smo->y2;
	double gamma;
	double L, H, slope, change;
	double k11, k12, k22, eta;
	
	if (smo->y1 == smo->y2)
		gamma = smo->alpha1 + smo->alpha2;
	else
		gamma = smo->alpha1 - smo->alpha2;
	if (s == 1)
	{
		L = MAX(0, gamma - C);
		H = MIN(C, gamma);
	} else {
		L = MAX(0, -1*gamma);
		H = MIN(C, C - gamma);
	}
	
	if (H <= L) {
		return 0;
	}
	
	k11 = kernel(X, dim, smo->I_up, smo->I_up, 'g', sigma);
	k22 = kernel(X, dim, smo->I_low, smo->I_low, 'g', sigma);
	k12 = kernel(X, dim, smo->I_up, smo->I_low, 'g', sigma);
	eta = 2*k12 - k11 - k22;
	
	if (eta < eps * (k11 + k22))
	{
		smo->a2 = smo->alpha2 - (smo->y2*(smo->F1 - smo->F2)/eta);
		if (smo->a2 < L)
			smo->a2 = L;
		else if (smo->a2 > H)
			smo->a2 = H;
	} else {
		slope = smo->y2*(smo->F1 - smo->F2);
		change = slope * (H - L);
		if (change != 0)
		{
			if (slope > 0)
				smo->a2 = H;
			else 
				smo->a2 = L;
		} else {
			smo->a2 = smo->alpha2;
		}
	}
	
	if (smo->a2 > C - eps * C) smo->a2 = C;
	else if (smo->a2 < eps * C) smo->a2 = 0;
	
	if (ABS(smo->a2 - smo->alpha2) < eps * (smo->a2 + smo->alpha2 + eps)) return 0;
	
	if (s == 1) smo->a1 = gamma - smo->a2;
	else smo->a1 = gamma + smo->a2;
	
	if (smo->a1 > C - eps * C) smo->a1 = C;
	else if (smo->a1 < eps * C) smo->a1 = 0;
	
	smo->Dual = smo->Dual - (smo->a1 - smo->alpha1) * (smo->F1 - smo->F2) / smo->y1 + 1 / 2 * eta * (smo->a1 - smo->alpha2) * (smo->a1 - smo->alpha2) / smo->y1 / smo->y1;
	return 1;
}

void getValueFromIndex(int index_type, double alphas[], double ylabel[], double Err[], SMO *smo, int index)
{
	PRINT("index: %d\n", index);
	if (index_type == I_UP) {
		smo->alpha1 = alphas[index];
		smo->y1 = ylabel[index];
		smo->F1 = Err[index];
	}
	else if (index_type == I_LOW) {
		smo->alpha2 = alphas[index];
		smo->y2 = ylabel[index];
		smo->F2 = Err[index];
	}
}

void updateToRoot(int rank, SMO *smo)
{
	if (rank != 0) {
		if (my_rank == rank)
			MPI_Send(smo, 1, smoType, 0, 0, MPI_COMM_WORLD);
		else if (my_rank == 0)
			MPI_Recv(smo, 1, smoType, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

void mergeLocalAlphas(double *local_alphas, double *alphas, int local_size, int size)
{
	int i, j, k = 0;

	if (my_rank == 0) {
		for (i = 0 ; i < size ; i += comm_sz)
			alphas[i] = local_alphas[k++];
	}

	for (i = 1 ; i < comm_sz ; i++) {
		k = 0;
		if (my_rank == 0) {
			MPI_Recv(local_alphas, local_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (j = i ; j < size ; j += comm_sz)
				alphas[j] = local_alphas[k++];
		}
		else
			MPI_Send(local_alphas, local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}

void updateParams(SMO *smo, SMO *local_smo, double *local_alphas, double *local_ylabel, double *local_Err)
{
	int i;
	SMO temp_smo;
	if (my_rank == 0) {
		smo->z1 = 0;
		smo->z2 = 0;
		smo->I_up = local_smo->I_up * comm_sz;
		smo->I_low = local_smo->I_low * comm_sz;
		smo->b_up = local_smo->b_up;
		smo->b_low = local_smo->b_low;
		for (i = 1; i < comm_sz; i++) {
			MPI_Recv(&temp_smo, 1, smoType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (temp_smo.b_up < smo->b_up) {
				smo->b_up = temp_smo.b_up;
				smo->I_up = temp_smo.I_up * comm_sz + i;
				smo->z1 = i;
			}
			if (temp_smo.b_low > smo->b_low) {
				smo->b_low = temp_smo.b_low;
				smo->I_low = temp_smo.I_low * comm_sz + i;
				smo->z2 = i;
			}
		}
	}
	else {
		MPI_Send(local_smo, 1, smoType, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Bcast(smo, 1, smoType, 0, MPI_COMM_WORLD);


	if (my_rank == smo->z1) 
		getValueFromIndex(I_UP, local_alphas, local_ylabel, local_Err, smo, local_smo->I_up);
	updateToRoot(smo->z1, smo);
	MPI_Bcast(smo, 1, smoType, 0, MPI_COMM_WORLD);

	if (my_rank == smo->z2)
		getValueFromIndex(I_LOW, local_alphas, local_ylabel, local_Err, smo, local_smo->I_low);
	updateToRoot(smo->z2, smo);
	MPI_Bcast(smo, 1, smoType, 0, MPI_COMM_WORLD);
}

void initMPI()
{
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	/* create MPI structure for SMO */
	int blockcounts[2];
	MPI_Aint offsets[2];
	MPI_Datatype oldTypes[2];
	MPI_Aint lb, int_extent;

	MPI_Type_get_extent(MPI_INT, &lb, &int_extent);

	blockcounts[0] = 6;
	blockcounts[1] = 11;
	offsets[0] = 0;
	offsets[1] = 6 * int_extent;
	oldTypes[0] = MPI_INT;
	oldTypes[1] = MPI_DOUBLE;

	MPI_Type_create_struct(2, blockcounts, offsets, oldTypes, &smoType);
	MPI_Type_commit(&smoType);
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
double* modified_SMO(double X[], int local_Y[], int local_size, int dim, double C, double sigma, double tau)
{
	int i;
	double local_ylabel[local_size];
	double local_b = 0.0;
	double* local_alphas;
	double local_Err[local_size];
	int numChanged;
	double DualityGap, local_DualityGap;
	SMO smo = { 0 }, local_smo = { 0 };

	local_alphas = (double *)malloc((local_size)*sizeof(double));
	
	/* initialize alpha, Err, Dual */
	for (i = 0; i < local_size; i++) {
		if (local_Y[i] == 0) local_ylabel[i] = -1;
		else local_ylabel[i] = 1;
		local_alphas[i] = 0.0;
		local_Err[i] = -1 * local_ylabel[i];
	}

	/* initialize b_up, I_up, b_low, I_low, DualityGap */
	local_DualityGap = computeDualityGap(local_Err, C, local_b, local_alphas, local_ylabel, local_size);
	computeBupIup(local_Err, C, local_alphas, local_ylabel, local_size, &local_smo.b_up, &local_smo.I_up);
	computeBlowIlow(local_Err, C, local_alphas, local_ylabel, local_size, &local_smo.b_low, &local_smo.I_low);

	MPI_Reduce(&local_DualityGap, &DualityGap, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&DualityGap, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	PRINT("----I_up: %d, I_low: %d, b_up: %lf, b_low: %lf----\n", local_smo.I_up, local_smo.I_low, local_smo.b_up, local_smo.b_low);

	
	PRINT("alpha1: %lf, y1: %d, F1: %lf\n", smo.alpha1, smo.y1, smo.F1);
	PRINT("alpha2: %lf, y2: %d, F2: %lf\n", smo.alpha2, smo.y2, smo.F2);


	numChanged = 1;

	while(DualityGap > tau*ABS(smo.Dual) && numChanged != 0)
	{
		updateParams(&smo, &local_smo, local_alphas, local_ylabel, local_Err);

		if (my_rank == 0) {
			numChanged = computeNumChaned(X, &smo, dim, C, sigma);
			printf("numchanged: %d\n", numChanged);
		}

		MPI_Bcast(&numChanged, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (my_rank == smo.z1) {
			smo.a1_old = local_alphas[local_smo.I_up];
			local_alphas[local_smo.I_up] = smo.a1;
		}
		updateToRoot(smo.z1, &smo);
		MPI_Bcast(&smo, 1, smoType, 0, MPI_COMM_WORLD);

		if (my_rank == smo.z2) {
			smo.a2_old = local_alphas[local_smo.I_low];
			local_alphas[local_smo.I_low] = smo.a2;
		}
		updateToRoot(smo.z2, &smo);
		MPI_Bcast(&smo, 1, smoType, 0, MPI_COMM_WORLD);

		PRINT("%d %d %d %d %lf %lf %lf %lf %d %d %lf %lf %lf %lf %lf %lf %lf\n", smo.I_up, smo.I_low, smo.z1, smo.z2, smo.b_up, smo.b_low, smo.alpha1, smo.alpha2, smo.y1, smo.y2, smo.F1, smo.F2, smo.a1, smo.a2, smo.a1_old, smo.a2_old, smo.Dual);

		/* update Err[i] */
		for (i = 0; i < local_size; i++) {
			local_Err[i] += (local_alphas[local_smo.I_up] - smo.a1_old) * local_ylabel[local_smo.I_up] * kernel(X, dim, smo.I_up, i, 'g', sigma) 
				+ (local_alphas[local_smo.I_low] - smo.a2_old) * local_ylabel[local_smo.I_low] * kernel(X, dim, smo.I_low, i, 'g', sigma);  
		}
		
		computeBupIup(local_Err, C, local_alphas, local_ylabel, local_size, &local_smo.b_up, &local_smo.I_up);
		computeBlowIlow(local_Err, C, local_alphas, local_ylabel, local_size, &local_smo.b_low, &local_smo.I_low);

		local_b = (local_smo.b_low + local_smo.b_up) / 2;
		local_DualityGap = computeDualityGap(local_Err, C, local_b, local_alphas, local_ylabel, local_size);

		MPI_Reduce(&local_DualityGap, &DualityGap, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(&DualityGap, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		PRINT("alpha1: %lf, y1: %d, F1: %lf, Gap: %lf\n", smo.alpha1, smo.y1, smo.F1, DualityGap);
		PRINT("alpha2: %lf, y2: %d, F2: %lf\n", smo.alpha2, smo.y2, smo.F2);
	}

	local_b = (local_smo.b_low + local_smo.b_up) / 2;
	local_DualityGap = computeDualityGap(local_Err, C, local_b, local_alphas, local_ylabel, local_size);

	MPI_Reduce(&local_DualityGap, &DualityGap, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&DualityGap, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//if (my_rank == comm_sz - 1)
	//	local_alphas[local_size-1] = local_b;



	return local_alphas;
}



int main(int argc, char* argv[])
{
	int size, local_size, dim, class;
	int i, j, k;
	double* x;
	int *y, *local_y, *local_temp_y;
	FILE *pFile, *pFile1; 
	double C;
	double sigma;
	double tau;
	double *alphas, *local_alphas;
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

	initMPI();
 		
	local_size = size / comm_sz;

	if (my_rank == 0) {
		y = (int *)malloc(size*sizeof(double));
		alphas = (double *)malloc((size+1)*sizeof(double));
	}

	/* all processes need a copy of x */
	x = (double *)malloc(size*dim*sizeof(double));
      	local_y = (int *)malloc((local_size)*sizeof(int));
	local_temp_y = (int *)malloc((local_size)*sizeof(int));
	local_alphas = (double *)malloc((local_size)*sizeof(double));


	pFile = fopen(argv[1], "r");
	if (pFile == NULL) {
		printf("can't open %s\n", argv[1]);
		exit(-1);
	}
	for (i = 0; i < size*dim; i++)
		fscanf(pFile, "%lf", &x[i]);

	if (my_rank == 0) {
		pFile = fopen(argv[2], "r");
		if (pFile == NULL) {
			printf("can't open %s\n", argv[2]);
			exit(-1);
		}
		for (i = 0; i < size; i++)
			fscanf(pFile, "%d", &y[i]);
		
	      	for (i = 1; i < comm_sz; i++) {
      			for (k = i, j = 0; k < size; k += comm_sz)
      				local_y[j++] = y[k];

			MPI_Send(local_y, local_size, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
		for (i = 0, j = 0; i < size; i += comm_sz)
			local_y[j++] = y[i];
      	}
      	else
		MPI_Recv(local_y, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	// for save the parameters 
	if (my_rank == 0) {
		pFile1 = fopen(filename, "w"); 
		if (pFile1 == NULL) {
			printf("can't open %s\n", filename);
			exit(-1);
		}
	}


	/* start the SMO algorithm */
	tau = 0.000001;

	GET_TIME(start);
	/* one-against-all method */
	for (i = 0; i < class; i++)
	{
		for (k = 0; k < local_size; k++) {
			if (local_y[k] == i) local_temp_y[k] = 1;
			else local_temp_y[k] = 0;
		}
		local_alphas = modified_SMO(x, local_temp_y, local_size, dim, C, sigma, tau);

		//mergeLocalAlphas(local_alphas, alphas, local_size, size);
		
		if (my_rank == 0) {
			b = -1 * alphas[size];
			/* save the result */
			for (k = 0; k < size; k++)
				fprintf(pFile1, "%lf\n", alphas[k]);		
		
			fprintf(pFile1, "%lf\n", b);
		}
	}
	GET_TIME(end);
	if (my_rank == 0)
		printf("The elapsed time is %e seconds\n", end - start);

	//free(x);
	//free(y);
	//fclose(pFile);
	//fclose(pFile1);
	return 0;
}
