#include <iostream>
#include <math.h>
#include <omp.h>
using namespace std;

float form_jacobi(float** alf, float* x, float* x1, float* bet, int n)
{
	int i, j;
	float s, max;
	for (i = 0; i < n; i++)
	{
		s = 0;
		for (j = 0; j < n; j++) {
			s += alf[i][j] * x[j];
		}
		s += bet[i];
		if (i == 0) {
			max = fabs(x[i] - s);
		}
		else if (fabs(x[i] - s) > max) {
			max = fabs(x[i] - s);
		}
		x1[i] = s;
	}
	return max;
}

float form_jacobi_parallel(float** alf, float* x, float* x1, float* bet, int n)
{
	int i, j;
	float s, max;
#pragma omp parallel for shared(alf, bet, x,x1) private (i,j,s)
	for (i = 0; i < n; i++)
	{
		s = 0;
		for (j = 0; j < n; j++) {
			s += alf[i][j] * x[j];
		}
		s += bet[i];
		if (i == 0) {
			max = fabs(x[i] - s);
		}
		else if (fabs(x[i] - s) > max) {
			max = fabs(x[i] - s);
		}
		x1[i] = s;
	}
	return max;
}

void parallelSolve(float** a, float* b, float* x, int n, float eps)
{
	float** f, * h, ** alf, * bet, * x1, * xx, max;
	int i, j, kvo;
	float t1, t2;
	f = new float* [n];
	for (i = 0; i < n; i++)
		f[i] = new float[n];
	h = new float[n];
	alf = new float* [n];
	for (i = 0; i < n; i++)
		alf[i] = new float[n];
	bet = new float[n];
	x1 = new float[n];
	xx = new float[n];

#pragma omp parallel for private (i,j)   
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
			if (i == j) alf[i][j] = 0; else alf[i][j] = -a[i][j] / a[i][i];
		bet[i] = b[i] / a[i][i];
	}
	for (i = 0; i < n; i++)
		x1[i] = bet[i];
	kvo = 0; max = 5 * eps;
	t1 = omp_get_wtime();
	while (max > eps)
	{

		for (i = 0; i < n; i++)
			x[i] = x1[i];
		max = form_jacobi_parallel(alf, x, x1, bet, n);
		kvo++;
	}
	t2 = omp_get_wtime();
	std::cout << "amount of iterations " << kvo << "\n";
}
void sequentialSolve(float** a, float* b, float* x, int n, float eps)
{
	float** f, * h, ** alf, * bet, * x1, * xx, max;
	int i, j, kvo;

	f = new float* [n];
	for (i = 0; i < n; i++) {
		f[i] = new float[n];
	}
	h = new float[n];
	alf = new float* [n];
	for (i = 0; i < n; i++) {
		alf[i] = new float[n];
	}
	bet = new float[n];
	x1 = new float[n];
	xx = new float[n];

	for (i = 0; i < n; bet[i] = b[i] / a[i][i], i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				alf[i][j] = 0;
			}
			else {
				alf[i][j] = -a[i][j] / a[i][i];
			}
		}
	}
	for (i = 0; i < n; i++) {
		x1[i] = bet[i];
	}
	kvo = 0;
	max = 5 * eps;

	while (max > eps)
	{
		for (i = 0; i < n; i++) {
			x[i] = x1[i];
		}
		max = form_jacobi(alf, x, x1, bet, n);
		kvo++;
	}
	std::cout << "amount of iterations: " << kvo << endl;
}

float form(float** alf, float* x, float* bet, int n)
{
	int i, j;
	float s, max;
	for (i = 0; i < n; i++)
	{
		s = 0;
		for (j = 0; j < n; j++) {
			s += alf[i][j] * x[j];
		}
		s += bet[i];
		if (i == 0) {
			max = fabs(x[i] - s);
		}
		else if (fabs(x[i] - s) > max) {
			max = fabs(x[i] - s);
		}
		x[i] = s;
	}
	return max;
}

float form_parallel(float** alf, float* x, float* bet, int n)
{
	int i, j;
	float s, max;
#pragma omp parallel for shared(alf, bet, x) private (i,j,s)
	for (i = 0; i < n; i++)
	{
		s = 0;
		for (j = 0; j < n; j++) {
			s += alf[i][j] * x[j];
		}
		s += bet[i];
		if (i == 0) {
			max = fabs(x[i] - s);
		}
		else if (fabs(x[i] - s) > max) {
			max = fabs(x[i] - s);
		}
		x[i] = s;
	}
	return max;
}

int main(int argc, char** argv)
{
	setlocale(LC_ALL, "Russian");
	int i, j, N;
	float** a, * b, * x, t1parl, t2parl, t1posl, t2posl, ep;
	std::cout << "Enter the matrix size >> ";
	cin >> N;
	ep = 1e-6;
	a = new float* [N];
	for (i = 0; i < N; i++) {
		a[i] = new float[N];
	}
	b = new float[N];
	x = new float[N];

	for (i = 0; i < N; a[i][i] = 1, i++) {
		for (j = 0; j < N; j++) {
			if (i != j) {
				a[i][j] = 0.1 / (i + j);
			}
		}
	}
	for (i = 0; i < N; i++)
		b[i] = sin(i);

	std::cout << "1. sequental algorithm " << endl;
	t1posl = omp_get_wtime();
	sequentialSolve(a, b, x, N, ep);
	t2posl = omp_get_wtime();
	std::cout << "vector x: " << x[0] << "\t" << x[N / 2] << "\t" << x[N - 1] << endl;
	std::cout << "time " << t2posl - t1posl << endl;
	std::cout << endl;

	std::cout << "2. parallel algorithm " << endl;
	t1parl = omp_get_wtime();
	parallelSolve(a, b, x, N, ep);
	t2parl = omp_get_wtime();
	std::cout << "vector X" << x[0] << "\t" << x[N / 2] << "\t" << x[N - 1] << endl;
	std::cout << "time " << t2parl - t1parl << endl;
	return 0;
}