#ifndef FOURIERSOLVERKERNALS_H
#define FOURIERSOLVERKERNALS_H

#include <helper_math.h>
#include <cuda_runtime.h>

struct FourierBuffers
{
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief buffer to store our histogram information
    //----------------------------------------------------------------------------------------------------------------------
    float *pdf;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief buffer to store our power spectrum information
    //----------------------------------------------------------------------------------------------------------------------
    float *ps;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief buffer to store our differencials
    //----------------------------------------------------------------------------------------------------------------------
    float2 *diffs;
    //----------------------------------------------------------------------------------------------------------------------
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief function to fill our histogram
/// @param _stream - our cuda stream to run our kernal on
/// @param _numThreads - number of threads we have per block
/// @param _numDiffs - total number of differencials
/// @param _res - resolution of our histogram
/// @param _buffs - our buffers of data for fourier analysis
//----------------------------------------------------------------------------------------------------------------------
void fillPDF(cudaStream_t _stream, int _numThreads, int _numDiffs, int _res, FourierBuffers _buffs);
//----------------------------------------------------------------------------------------------------------------------
/// @brief function to call analysis of our information
/// @param _stream - our cuda stream to run our kernal on
/// @param _numThreads - number of threads we have per block
/// @param _numDiffs - total number of differencials
/// @param _res - resolution of our histogram
/// @param _SDSqrd - standard deviation squared. Used in our guassian function.
/// @param _buffs - our buffers of data for fourier analysis
//----------------------------------------------------------------------------------------------------------------------
void analyse(cudaStream_t _stream, int _numThreads, int _numDiffs, int _res, float _SDSqrd, FourierBuffers _buffs);
//----------------------------------------------------------------------------------------------------------------------

#endif // FOURIERSOLVERKERNALS_H

