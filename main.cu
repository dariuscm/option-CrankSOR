#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "CrankNicolsonSOR.cuh"
#include <iostream>
#include <Windows.h>
#include <vector>

using namespace std;

int main()
{
	
	cudaDeviceProp data;
	cudaGetDeviceProperties(&data, 0);
	cout << "Device name: " << data.name << endl;
	cout << data.maxThreadsPerBlock << " - MAX THREADS PER BLOCK" << endl;
	cout << data.sharedMemPerBlock << " - SHARED MEM PER BLOCK" << endl;
	

	double s0_ = 100.0, r_ = 0.05, sigma_ = 0.2;
	double K_ = 100.0, T_ = 1., sl_ = 0.0, su_ = 200.0, omega = 1.;
	const int imax = 500, jmax = 500, nmax = 50;

	BSPDE PDE(s0_, r_, sigma_, K_, T_, sl_, su_, imax, jmax);

	double V[jmax + 1];

	for (int j = 0; j <= jmax; j++)
	{
		V[j] = PDE.f_h(j);
	}

	double *d_V;
	BSPDE *d_PDE;

	cudaMalloc(&d_PDE, sizeof(PDE));
	cudaMemcpy(d_PDE, &PDE, sizeof(PDE), cudaMemcpyHostToDevice);

	double *data;
	cudaMalloc(&data, sizeof(double) * imax*(jmax+1));

	cudaMalloc(&d_V, sizeof(V));
	cudaMemcpy(d_V, &V, sizeof(V), cudaMemcpyHostToDevice);

	SOR <<<1, jmax - 1 >>>(d_V, d_PDE, omega, nmax, data);

	double *dataHost;
	dataHost = new double [imax*(jmax + 1)];
	cudaMemcpy(dataHost, data, sizeof(double) * imax*(jmax + 1), cudaMemcpyDeviceToHost);

	vector<vector<double>> dataVector;
	dataVector.resize(imax + 1);
	for (int i = 0; i <= imax;i++)
		dataVector[i].resize(jmax + 1);
	for (int j = 0; j <= jmax;j++)
		dataVector[imax][j] = PDE.f_h(j);
	for (int i = 0; i < imax; i++)
		for (int j = 0; j <= jmax; j++)
			dataVector[i][j] = dataHost[i*jmax + j];

	delete [] dataHost;
	cudaMemcpy(&V, d_V, sizeof(V), cudaMemcpyDeviceToHost);
	
	double price;
	double *d_price;

	cudaMalloc(&d_price, sizeof(double));
	
	Price<<<1, 1>>>(d_price, d_V, d_PDE, s0_);

	cudaMemcpy(&price, d_price, sizeof(double), cudaMemcpyDeviceToHost);
	cout << "Price: " << price << endl << endl;

	cudaFree(d_V);
	cudaFree(d_PDE);
	cudaFree(d_price);

	system("pause");
	return 0;
}