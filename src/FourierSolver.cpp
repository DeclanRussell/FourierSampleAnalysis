#include "FourierSolver.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <iostream>
#include <QColor>

//----------------------------------------------------------------------------------------------------------------------
FourierSolver::FourierSolver() : m_resolution(110),m_rangeSelection(0.1f)
{
    setStandardDeviation(0.02f);
    m_psImage = QImage(m_resolution,m_resolution,QImage::Format_RGB32);
    m_pdf.resize(m_resolution*m_resolution);

    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found" << count << "CUDA device(s)" << std::endl;
    if(count == 0){
        std::cerr<<"Install an Nvidia chip!"<<std::endl;
        return;
    }
    for (int i=0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout<<prop.name<<", Compute capability:"<<prop.major<<"."<<prop.minor<<std::endl;;
        std::cout<<"  Global mem: "<<prop.totalGlobalMem/ 1024 / 1024<<"M, Shared mem per block: "<<prop.sharedMemPerBlock / 1024<<"k, Registers per block: "<<prop.regsPerBlock<<std::endl;
        std::cout<<"  Warp size: "<<prop.warpSize<<" threads, Max threads per block: "<<prop.maxThreadsPerBlock<<", Multiprocessor count: "<<prop.multiProcessorCount<<" MaxBlocks: "<<prop.maxGridSize[0]<<std::endl;
        m_maxNumThreads = prop.maxThreadsPerBlock;
    }

    // Create our cuda stream
    checkCudaErrors(cudaStreamCreate(&m_stream));

    m_deviceBuffers.diffs = 0;
    m_deviceBuffers.pdf = 0;
    m_deviceBuffers.ps = 0;

}
//----------------------------------------------------------------------------------------------------------------------
FourierSolver::~FourierSolver()
{   
    // Delete our CUDA streams as well
    checkCudaErrors(cudaStreamDestroy(m_stream));
    // Delete our CUDA buffers fi they have anything in them
    if(m_deviceBuffers.diffs) checkCudaErrors(cudaFree(m_deviceBuffers.diffs));
    if(m_deviceBuffers.pdf) checkCudaErrors(cudaFree(m_deviceBuffers.pdf));
    if(m_deviceBuffers.ps) checkCudaErrors(cudaFree(m_deviceBuffers.ps));
    m_deviceBuffers.diffs = 0;
    m_deviceBuffers.pdf = 0;
    m_deviceBuffers.ps = 0;
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::import2DFromFile(QString _dir)
{
    // Delete any data we may have already imported
    m_2DPoints.clear();
    m_differentials.clear();
    // Read our 2D points from our file into our array
    std::ifstream file(_dir.toStdString());
    float2 p;
    float2 min,max;
    bool mmSet = false;
    if(file.is_open())
    {
        while (!file.eof()) {
            file >> p.x;
            file >> p.y;
            m_2DPoints.push_back(p);
            if(mmSet)
            {
                if(p.x<min.x) min.x = p.x;
                if(p.y<min.y) min.y = p.y;
                if(p.x>max.x) max.x = p.x;
                if(p.y>max.y) max.y = p.y;
            }
            else
            {
                min = max = p;
                mmSet = true;
            }
        }
        file.close();
        std::cout<<"Total of "<<m_2DPoints.size()<<" points"<<std::endl;
    }
    else
    {
        std::cerr<<"Could not open file :("<<std::endl;
    }
    // Move all our points to between 0 and 1 for ease of use
    for(unsigned int i=0; i<m_2DPoints.size(); i++)
    {
        m_2DPoints[i]-=min;
        m_2DPoints[i]/=max - min;
    }

    // Calculate our pair-wise differencials. This is a generalisation of the fourier transform to improve performance.
    // We will also create our probability density histogram here

    mmSet = false;
    for(unsigned int i=0; i<m_2DPoints.size();i++)
    for(unsigned int j=0;j<m_2DPoints.size();j++)
    {
        if(i==j) continue;
        p = m_2DPoints[i]-m_2DPoints[j];
        if(fabs(p.x)>m_rangeSelection||fabs(p.y)>m_rangeSelection) continue;
        m_differentials.push_back(p);
        if(mmSet)
        {
            if(p.x<min.x) min.x = p.x;
            if(p.y<min.y) min.y = p.y;
            if(p.x>max.x) max.x = p.x;
            if(p.y>max.y) max.y = p.y;
        }
        else
        {
            min = max = p;
            mmSet = true;
        }

    }

    std::cout<<"num diffs "<<m_differentials.size()<<std::endl;

    // Move all our differentials to between -1 and 1 for ease of use
    for(unsigned int i=0; i<m_differentials.size(); i++)
    {
        m_differentials[i]-=min;
        m_differentials[i]/=max - min;
        m_differentials[i]*=2.0f;
        m_differentials[i]-=1.0f;
    }

}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::importDifferentialsFromFile(QString _dir)
{
    m_differentials.clear();
    // Read our 2D points from our file into our array
    std::ifstream file(_dir.toStdString());
    float2 p;
    float2 min,max;
    bool mmSet = false;
    if(file.is_open())
    {
        while (!file.eof()) {
            file >> p.x;
            file >> p.y;
            if(fabs(p.x)>m_rangeSelection||fabs(p.y)>m_rangeSelection) continue;
            m_differentials.push_back(p);
            if(mmSet)
            {
                if(p.x<min.x) min.x = p.x;
                if(p.y<min.y) min.y = p.y;
                if(p.x>max.x) max.x = p.x;
                if(p.y>max.y) max.y = p.y;
            }
            else
            {
                min = max = p;
                mmSet = true;
            }
        }
        file.close();
        std::cout<<"Total of "<<m_differentials.size()<<" differentials"<<std::endl;
    }
    else
    {
        std::cerr<<"Could not open file :("<<std::endl;
    }

    // Move all our differentials to between -1 and 1 for ease of use
    for(unsigned int i=0; i<m_differentials.size(); i++)
    {
        m_differentials[i]-=min;
        m_differentials[i]/=max - min;
        m_differentials[i]*=2.0f;
        m_differentials[i]-=1.0f;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::analysePoints()
{

    // Load our differencials to our device
    // Delete our CUDA buffers fi they have anything in them
    if(m_deviceBuffers.diffs) checkCudaErrors(cudaFree(m_deviceBuffers.diffs));
    if(m_deviceBuffers.pdf) checkCudaErrors(cudaFree(m_deviceBuffers.pdf));
    if(m_deviceBuffers.ps) checkCudaErrors(cudaFree(m_deviceBuffers.ps));
    m_deviceBuffers.diffs = 0;
    m_deviceBuffers.pdf = 0;
    m_deviceBuffers.ps = 0;
    // Send the data to the GPU
    checkCudaErrors(cudaMalloc(&m_deviceBuffers.diffs,m_differentials.size()*sizeof(float2)));
    checkCudaErrors(cudaMemcpy(m_deviceBuffers.diffs,&m_differentials[0],sizeof(float2)*m_differentials.size(),cudaMemcpyHostToDevice));
    // Allocate the space for our histogram and power spectrum
    checkCudaErrors(cudaMalloc(&m_deviceBuffers.pdf,m_resolution*m_resolution*sizeof(float)));
    checkCudaErrors(cudaMalloc(&m_deviceBuffers.ps,m_resolution*m_resolution*sizeof(float)));

    // Fill our pdf
    fillPDF(m_stream,m_maxNumThreads,(int)m_differentials.size(),m_resolution,m_deviceBuffers);

    // Perfrom our analysis
    analyse(m_stream,m_maxNumThreads,(int)m_differentials.size(),m_resolution,m_stanDevSqrd,m_deviceBuffers);


//    // Calculate our pair-wise differencials within our range selection
//    float2 p;
//    float prop = 1.f/(float)m_differentials.size();
//    int idx;
//    for(unsigned int i=0; i<m_differentials.size();i++)
//    {
//        p = m_differentials[i];

//        // Build up our pfd histogram
//        p+=1.0f;
//        p/=2.0f;
//        p*=m_resolution;
//        idx = (int)floor(p.x)+(int)floor(p.y)*m_resolution;
//        if(idx<m_pdf.size()){
//            m_pdf[(int)floor(p.x)+(int)floor(p.y)*m_resolution]+=prop;
//        }
//        else
//        {
//            std::cout<<"idx "<<idx<<" d "<<m_differentials[i].x<<","<<m_differentials[i].y<<std::endl;
//        }
//    }

#ifdef USE_PTHREADS
    std::vector<pthread_t> pid;
    int cellLeftToCompute = m_psImage.width()*m_psImage.height();
    int blocks = ceil(cellLeftToCompute/MAX_THREADS);
    pid.resize(MAX_THREADS);
    std::vector<PSAArgs> a;
    a.resize(MAX_THREADS);
    int rc;
    int numThreads = MAX_THREADS;
    for(int i=0;i<blocks;i++)
    {
        if(cellLeftToCompute<numThreads) numThreads = cellLeftToCompute;
        for(int j=0;j<numThreads;j++)
        {

            int idx = i*MAX_THREADS+j;
            a[j].obj = this;
            a[j].x = floor(idx/m_psImage.width());
            a[j].y = idx-(a[j].x*m_psImage.width());

            rc = pthread_create(&pid[j],NULL,psaWrapper,(void*)&a[j]);
            if (rc)
            {
                std::cout<<" failed on thread "<<idx<<std::endl;
                exit(-1);
                std::cout << "Error:unable to create thread," << rc << std::endl;
            }
        }

        void *status;
        //Wait for all the treads to finish
        for(int k=0;k<numThreads;k++)
        {
              rc = pthread_join(pid[k], &status);
              if (rc){
                 std::cout << "Error:unable to join," << rc << std::endl;
              }
        }
        std::cout<<((float)i/(float)(blocks-1))*100.f<<"% Complete!"<<std::endl;
    }

    std::cout<<"Threads complete!"<<std::endl;

    // Can parallise these next bits later
    float max = 0;
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
        if(m_ps[x][y]>max) max = m_ps[x][y];
    }
    float ps;
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
        ps = m_ps[x][y]/max;
        ps*=255;
        m_psImage.setPixel(x,y,QColor(ps,ps,ps).rgb());
    }

//#else
    float ps;
    float2 freqVector;
    float max = 0;
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
        freqVector = float2((float)(x-m_psImage.width()*.5f)/m_psImage.width(),(float)(y-m_psImage.height()*.5f)/m_psImage.height());
        freqVector *= m_axisRange;
        ps = 0.f;
        for(unsigned int i=0;i<m_sampleDiff.size();i++)
        {
            ps+= gausian(freqVector,m_sampleDiff[i])*pdf(m_sampleDiff[i]);
        }
        ps*=m_sampleDiff.size();

        if(ps>max) max = ps;
        //ps = m_pdf[x][y]*m_sampleDiff.size();
        std::cout<<"ps" <<ps<<std::endl;
        //if(ps>255)ps=255;
        m_ps[x][y] = ps;
        //m_psImage.setPixel(x,y,QColor(ps,ps,ps).rgb());
    }
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
        ps = m_ps[x][y]/max;
        ps*=255;
        m_psImage.setPixel(x,y,QColor(ps,ps,ps).rgb());
    }
#endif

    //Copy our data back to our host
    checkCudaErrors(cudaMemcpy(&m_pdf[0],m_deviceBuffers.ps,sizeof(float)*m_resolution*m_resolution,cudaMemcpyDeviceToHost));

    float ps;
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
        ps = m_pdf[x+y*m_resolution]*255;
        //if(ps>0) std::cout<<"ps "<<ps<<std::endl;
        if(ps>255)
        {
            ps=255;
        }
        m_psImage.setPixel(x,y,QColor((int)ps,(int)ps,(int)ps).rgb());
    }

}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::setResolution(int _r)
{
    m_resolution = _r;
    m_psImage = QImage(m_resolution,m_resolution,QImage::Format_RGB32);
    m_pdf.resize(m_resolution*m_resolution);
}
//----------------------------------------------------------------------------------------------------------------------
#ifdef USE_PTHREADS
void FourierSolver::perSampleAnalysis(int _x, int _y)
{
    //std::cout<<"x "<<_x<<" y "<<_y<<std::endl;
    float ps;
    float2 freqVector;
    freqVector = float2((float)(_x-m_psImage.width()*.5f)/m_psImage.width(),(float)(_y-m_psImage.height()*.5f)/m_psImage.height());
    freqVector *= m_axisRange;
    ps = 0.f;
    for(unsigned int i=0;i<m_sampleDiff.size();i++)
    {
        ps+= gausian(freqVector,m_sampleDiff[i])*pdf(m_sampleDiff[i]);
    }
    ps*=m_sampleDiff.size();

    //ps = m_pdf[x][y]*m_sampleDiff.size();
    //std::cout<<"ps" <<ps<<std::endl;
    //if(ps>255)ps=255;
    m_ps[_x][_y] = ps;
}
#endif
//----------------------------------------------------------------------------------------------------------------------
float FourierSolver::gausian(float2 _q, float2 _d)
{
    float2 sq = (_q-_d);
    sq*=sq;
    float w = pow(M_E,-((sq.x+sq.y)/(2.f*m_stanDevSqrd)));
    return w;
}
//----------------------------------------------------------------------------------------------------------------------
