////////////////////////////////////////////////////////////
//
//	BSPDE	- holds upper/lower/terminal boundary conditions
//			- holds BS + Option parameters
//			- a,b,c,d = BSEq co-efficients
//			- t(i), S(j) = time, stock-price at i,j
//			- A,B,C,D = finite difference co-efficients
//			- fl, fu = lower and upper boundary conditions
//			- f_h = host terminal boundary condition
//			
//			- SOR = Solves for t=0 present value vector
//			- Price = interpolates t=0 vector for price
//
////////////////////////////////////////////////////////////

#pragma once

#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>

class BSPDE
{
public:
	BSPDE(double s0_, double r_, double sigma_, double K_, double T_, double sl_, double su_, int imax_, int jmax_);
	__device__ double a(double t, double S);
	__device__ double b(double t, double S);
	__device__ double c(double t, double S);

	__device__ double t(double i);
	__device__ double S(int j);

	__device__ double fl(int i);
	__device__ double fu(int i);

	__host__ double f_h(int j);

	__device__ double A(int i, int j);
	__device__ double B(int i, int j);
	__device__ double C(int i, int j);
	__device__ double E(int i, int j);
	__device__ double F(int i, int j);
	__device__ double G(int i, int j);

	double s0, r, sigma, K, T, sl, su, dt, dS;
	int imax, jmax;

};

__global__ void SOR(double * V, BSPDE *PDE, double omega, double nmax, double * data);
__global__ void Price(double * returnPtr, double * V, BSPDE* PDE, double s0);