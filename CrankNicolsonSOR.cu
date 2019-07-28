#include "CrankNicolsonSOR.cuh"
#include <cmath>
#include <stdio.h>

BSPDE::BSPDE(double s0_, double r_, double sigma_, double K_, double T_, double sl_, double su_, int imax_, int jmax_) : s0(s0_), r(r_), sigma(sigma_), K(K_), T(T_), sl(sl_), imax(imax_), jmax(jmax_), su(su_)
{
	dt = T_ / imax; 
	dS = (su_ - sl_) / jmax;
}

__device__ double BSPDE::a(double t, double S)
{
	return -0.5*pow(sigma*S, 2.0);
}

__device__ double BSPDE::b(double t, double S)
{
	return -r*S;
}

__device__ double BSPDE::c(double t, double S)
{
	return r;
}

__device__ double BSPDE::t(double i)
{
	return i*dt;
}

__device__ double BSPDE::S(int j)
{
	return sl + j*dS;
}

__device__ double BSPDE::fl(int i)
{
	return K*exp(-r*(T - t(i)));
}

__device__ double BSPDE::fu(int i)
{
	return 0.0;
}

__device__ double BSPDE::A(int i, int j)
{
	return 0.5*dt*(b(t(i - 0.5), S(j)) / 2.0 - a(t(i - 0.5), S(j)) / dS) / dS;
}

__device__ double BSPDE::B(int i, int j)
{
	return 1.0 + 0.5*dt*(2.0*a(t(i - 0.5), S(j)) / (dS*dS) - c(t(i - 0.5), S(j)));
}

__device__ double BSPDE::C(int i, int j)
{
	return -0.5*dt*(b(t(i - 0.5), S(j)) / 2.0 + a(t(i - 0.5), S(j)) / dS) / dS;
}


__device__ double BSPDE::E(int i, int j)
{
	return -A(i, j);
}

__device__ double BSPDE::F(int i, int j)
{
	return 2.0 - B(i, j);
}

__device__ double BSPDE::G(int i, int j)
{
	return -C(i, j);
}

double BSPDE::f_h(int j)
{
	if (K > (sl + j*dS))
		return K - (sl + j*dS);
	else
		return 0;
}

__global__ void SOR(double * V, BSPDE *PDE, double omega, double nmax, double * data)
{
	
	int index = threadIdx.x + 1;

	__shared__ double V_[1026], V_Minus1[1026], P[1026], w_[1026], y[1026];

	if (threadIdx.x == 0)
	{
		V_[0] = V[0];
		V_[PDE->jmax] = V[PDE->jmax];
	}
	V_[index] = V[index];
	__syncthreads();

	for (int i = PDE->imax; i >= 1; i--)
	{
		V_Minus1[index] = V_[index];
		if (threadIdx.x == 0)
		{
			V_Minus1[0] = PDE->fl(i - 1);
			V_Minus1[PDE->jmax] = PDE->fu(i - 1);
		}	
		w_[index] = 0.0;
		if (index == 1)
			w_[index] = PDE->A(i, 1)*PDE->fl(i) - PDE->E(i, 1)*PDE->fl(i - 1);
		if (index == PDE->jmax - 1)
			w_[index] = PDE->C(i, PDE->jmax - 1)*PDE->fu(i) - PDE->G(i, PDE->jmax - 1)*PDE->fu(i - 1);

		P[index] = PDE->A(i, index)*V_[index - 1] + PDE->B(i, index)*V_[index] + PDE->C(i, index)*V_[index + 1];
		if(index == 1)
			P[index] = PDE->B(i, 1)*V_[1] + PDE->C(i, 1)*V_[2];
		if(index == PDE->jmax-1)
			P[index] = PDE->A(i, PDE->jmax - 1)*V_[PDE->jmax - 2] + PDE->B(i, PDE->jmax - 1)*V_[PDE->jmax - 1];
		__syncthreads();

		P[index] += w_[index];
		__syncthreads();

		for (int n = 0; n < nmax; n++)
		{
			y[index] = 1.0 / PDE->F(i, index) * (P[index] - PDE->E(i, index)*V_Minus1[index - 1] - PDE->G(i, index)*V_Minus1[index + 1]);
			__syncthreads();
			V_Minus1[index] += omega * (y[index] - V_Minus1[index]);
			__syncthreads();
		}
		V_[index] = V_Minus1[index];
		data[(i - 1)*PDE->jmax + index] = P[index];
		__syncthreads();
	}
	V[index] = V_[index];
}

__global__ void Price(double * returnPtr, double * V, BSPDE* PDE, double s0)
{
	int j = floor(((s0 - PDE->sl)*PDE->jmax) / (PDE->su - PDE->sl));
	double PCT = (s0 - PDE->S(j)) / (PDE->S(j + 1) - PDE->S(j));
	*returnPtr = V[j] + PCT*(V[j + 1] - V[j]);
}