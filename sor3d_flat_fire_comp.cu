#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 512

extern "C" {
__global__ void lagrangeUpdate(int ie, int je, int k, int klocal, \
         float orelax, float va2s, int dim_1,  int dim_2, int dim_3, int tnz, \
         float* ve,float* vf,float* vg,float* vh,float* vm,float* vn, \
         float* va2f,float* vr,float* vd,float* vp1,float* t_p1)
        {
	 int i = blockDim.x * blockIdx.x + threadIdx.x;
	 int j = blockDim.y * blockIdx.y + threadIdx.y;

	 int ijk=k*(dim_1*dim_2)+j*dim_1+i  ;
         int ip1jk=k*(dim_1*dim_2)+j*dim_1+(i+1)    ;
         int im1jk=k*(dim_1*dim_2)+j*dim_1+(i-1)    ;
         int ijp1k=k*(dim_1*dim_2)+(j+1)*dim_1+i    ;
         int ijm1k=k*(dim_1*dim_2)+(j-1)*dim_1+i    ;
         int ijkp1=(k+1)*(dim_1*dim_2)+j*dim_1+i    ;
         int ijkm1=(k-1)*(dim_1*dim_2)+j*dim_1+i    ;
         int ijkl=klocal*(dim_1*dim_2)+j*dim_1+i    ;

	 if (i < ie && j < je){
                t_p1[ijkl]=vd[ijk]*(ve[ijk]*vp1[ip1jk]+\
                      vf[ijk]*vp1[im1jk]+            \
                      vg[ijk]*vp1[ijp1k]+            \
                      vh[ijk]*vp1[ijm1k]+            \
                      (vm[ijk]*vp1[ijkp1]+           \
                      vn[ijk]*vp1[ijkm1])*           \
                      va2s / (va2f[ijk]*va2f[ijk])-  \
                      vr[ijk])+ orelax*vp1[ijk] ;
		}
	}


void sor3d_flat_fire_comp( int is, int ie, int js, \
	 int je, int k, int klocal, int step, \
         float orelax,  int dim_1,  int dim_2, int dim_3, int tnz, \
         float* ve,float* vf,float* vg,float* vh,float* vm,float* vn, \
	 float va2s,float* va2f,float* vr,float* vd,float* vp1,float* t_p1)
	{

	js=js-1;	
	is=is-1;	
	k=k-1;	
	klocal=klocal-1;	

	int size=dim_1*dim_2*dim_3*sizeof(float);
	int sizem1=dim_1*dim_2*(dim_3-1)*sizeof(float);
	int sizet1=dim_1*dim_2*tnz*sizeof(float);

	float *dd; 
	float *de; 
	float *df; 
	float *dg; 
	float *dh; 
	float *dm; 
	float *dn; 
	float *da2f; 
	float *dr; 
	float *dp1; 
	float *dtp1; 

	cudaMalloc((void **)&dd, sizem1);
	cudaMalloc((void **)&de, sizem1);
	cudaMalloc((void **)&df, sizem1);
	cudaMalloc((void **)&dg, sizem1);
	cudaMalloc((void **)&dh, sizem1);
	cudaMalloc((void **)&dm, sizem1);
	cudaMalloc((void **)&dn, sizem1);
	cudaMalloc((void **)&da2f, size);
	cudaMalloc((void **)&dr, sizem1);
	cudaMalloc((void **)&dp1, size);
	cudaMalloc((void **)&dtp1, sizet1);



	cudaMemcpy(dd, vd, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(de, ve, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(df, vf, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(dg, vg, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(dh, vh, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(dm, vm, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(dn, vn, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(dr, vr, sizem1, cudaMemcpyHostToDevice);
	cudaMemcpy(da2f, va2f, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dp1, vp1, size, cudaMemcpyHostToDevice);

	dim3 dblock(BLOCK_DIM,BLOCK_DIM);
  	dim3 dgrid(dim_1 / dblock.x, dim_2 / dblock.y);

	lagrangeUpdate<<<dgrid, dblock>>>(ie,je,k,klocal,orelax,va2s,dim_1,dim_2,dim_3,tnz,de,df,dg,dh,dm,dn,da2f,dr,dd,dp1,dtp1);

	cudaMemcpy(t_p1, dtp1, size, cudaMemcpyDeviceToHost);
}

}
