#include <stdio.h>

void sor3d_flat_fire_comp( int is, int ie, int js, \
	 int je, int k, int klocal, int step, \
         float orelax,  int dim1,  int dim2, int dim3, float* tnz, \
         float* ve,float* vf,float* vg,float* vh,float* vm,float* vn, \
	 float va2s,float* va2f,float* vr,float* vd,float* vp1,float* t_p1)
	{

	int i,j;

	js=js-1;	
	is=is-1;	
	k=k-1;	
	klocal=klocal-1;	


        for(j=js; j<je; j++){
             for(i=is; i<ie; i++){
		int ijk=k*(dim1*dim2)+j*dim1+i	;
		int ip1jk=k*(dim1*dim2)+j*dim1+(i+1)	;
		int im1jk=k*(dim1*dim2)+j*dim1+(i-1)	;
		int ijp1k=k*(dim1*dim2)+(j+1)*dim1+i	;
		int ijm1k=k*(dim1*dim2)+(j-1)*dim1+i	;
		int ijkp1=(k+1)*(dim1*dim2)+j*dim1+i	;
		int ijkm1=(k-1)*(dim1*dim2)+j*dim1+i	;
		int ijkl=klocal*(dim1*dim2)+j*dim1+i	;
                t_p1[ijkl]=vd[ijk]*(ve[ijk]*vp1[ip1jk]+\
                      vf[ijk]*vp1[im1jk]+            \
                      vg[ijk]*vp1[ijp1k]+            \
                      vh[ijk]*vp1[ijm1k]+            \
                      (vm[ijk]*vp1[ijkp1]+           \
                      vn[ijk]*vp1[ijkm1])* 	     \
                      va2s / (va2f[ijk]*va2f[ijk])-  \
                      vr[ijk])+ orelax*vp1[ijk] ;
	     }
	}
}

