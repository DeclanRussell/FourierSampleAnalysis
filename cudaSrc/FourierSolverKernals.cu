//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolverCUDAKernals.cu
/// @author Declan Russell
/// @date 03/02/2016
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include "FourierSolverKernals.h"
#include <helper_math.h>  //< some math operations with cuda types
#include <iostream>
#define M_E        2.71828182845904523536f

//----------------------------------------------------------------------------------------------------------------------
__global__ void fillZero(int _size, float *_buff)
{
    int idx = threadIdx.x + __mul24(blockIdx.x ,blockDim.x);
    if(idx<_size)
    {
        _buff[idx] = 0.f;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fillPDFKernel(int _numDiffs, int _pdfSize,int _res, float _invNumDiffs, FourierBuffers _buffs)
{
    int idx = threadIdx.x + blockIdx.x *blockDim.x;
    if(idx<_numDiffs)
    {
        float2 p = _buffs.diffs[idx];
        // Build up our pfd histogram
        p+=1.0f;
        p/=2.0f;
        p*=_res;
        int pdfIdx = (int)floor(p.x)+(int)floor(p.y)*_res;
        if(pdfIdx<_pdfSize){
            atomicAdd(&(_buffs.pdf[pdfIdx]), _invNumDiffs);
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float gausian(float2 _q, float2 _d, float _SDSqrd)
{
    float2 sq = (_q-_d);
    sq*=sq;
    float w = pow(M_E,-((sq.x+sq.y)/(2.f*_SDSqrd)));
    return w;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void analyseKernal(int _numDiffs, int _psSize, int _res, float _SDSqrd, FourierBuffers _buffs)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_psSize)
    {
        float2 freqVector;
        freqVector.y = floorf(idx/_res);
        freqVector.x = idx - (freqVector.y*_res);
        freqVector/=_res;
        freqVector*=2.f;
        freqVector-=1.f;

        int pdfIdx;
        float2 diff;
        float2 diffNorm;
        float ps = 0.f;
        for(int i=0;i<_numDiffs;i++)
        {
            diff = diffNorm = _buffs.diffs[i];
            diffNorm+=1.f;
            diffNorm/=2.f;
            diffNorm*=_res;
            pdfIdx = (int)floor(diffNorm.x)+(int)floor(diffNorm.y)*_res;
            ps+=gausian(freqVector,diff,_SDSqrd)*_buffs.pdf[pdfIdx];
        }

//        Put our result into our buffer
        _buffs.ps[idx] = ps;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void fillPDF(cudaStream_t _stream, int _numThreads, int _numDiffs, int _res, FourierBuffers _buffs)
{
    int blocks = 1;
    int threads = _numDiffs;
    if(_numDiffs>_numThreads)
    {
        //calculate how many blocks we want
        blocks = ceil(_numDiffs/_numThreads)+1;
        threads = _numThreads;
    }

    // Init our histogram to 0's
    fillZero<<<blocks,threads,0,_stream>>>(_res*_res,_buffs.pdf);

    //make sure all our threads are done
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Fill zero error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    // Fill our pdf
    fillPDFKernel<<<blocks,threads,0,_stream>>>(_numDiffs, _res*_res, _res, 1.f/(float)_numDiffs, _buffs);

    //make sure all our threads are done
    cudaThreadSynchronize();

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Fill PDF error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

}
//----------------------------------------------------------------------------------------------------------------------
void analyse(cudaStream_t _stream, int _numThreads, int _numDiffs, int _res, float _SDSqrd, FourierBuffers _buffs)
{
    int blocks = 1;
    int pdfSize = _res*_res;
    int threads = pdfSize;
    if(pdfSize>_numThreads)
    {
        //calculate how many blocks we want
        blocks = ceil(pdfSize/_numThreads)+1;
        threads = _numThreads;
    }

    //Perform our analysis
    analyseKernal<<<blocks,threads,0,_stream>>>(_numDiffs,_res*_res, _res,_SDSqrd, _buffs);

    //make sure all our threads are done
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Analyse error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

}
//----------------------------------------------------------------------------------------------------------------------
