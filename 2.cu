#include <iostream>
#include <stdlib.h>
#include <math.h>
#define BLK_SIZE 16
using namespace std;

__global__ void gpuMM(double *a,double *b, double *c, int N)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	double sum=0.0;
	if(row<N && col < N)
	{
		for(int i=0;i<N;i++)
			sum+=a[row*N+i]*b[i*N+col];
		c[row*N+col]=sum;
	}
	else
		return;
}

int main()
{
	int N,i,j,k;
	double *hA,*hB,*hC,*dA,*dB,*dC;

	cout<<"Enter N: ";
	cin>>N;


	hA = new double[N*N];
	hB = new double[N*N];
	hC = new double[N*N];
	int size = sizeof(double)*N*N;
	cudaMalloc(&dA,size);
	cudaMalloc(&dB,size);
	cudaMalloc(&dC,size);


	for(i=0;i<N*N;i++)
	{
		hA[i] = i;
		hB[i] = N*N -1;
	}

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threads_per_block(BLK_SIZE,BLK_SIZE);
	dim3 no_of_blocks(ceil((float)N/BLK_SIZE),ceil((float)N/BLK_SIZE));

	cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
	cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice);

	
	cudaEventRecord(start);	
	gpuMM<<<no_of_blocks,threads_per_block>>>(dA,dB,dC,N); /* function call for gpu action(gpuMM)(function call) */
	cudaEventRecord(stop);

    cudaMemcpy(hC,dC,size,cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds,start,stop);
	

	double *cc = new double[N*N];
	double sum=0.0;

	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			sum=0.0;
			for(k=0;k<N;k++)
			{
				sum+= hA[i*N+k]*hB[k*N+j];
			}
			
			cc[i*N+j]=sum;
			if(hC[i*N+j] != cc[i*N+j])
			{
				cout<<"Incorrect Result\n";
				exit(0);
			}

		}
	}


	cout<<"Correct Result time: "<<milliseconds/1000<<endl;
	free(hA);free(hB);free(hC);free(cc);
	cudaFree(dA);cudaFree(dB);cudaFree(dC);
}
