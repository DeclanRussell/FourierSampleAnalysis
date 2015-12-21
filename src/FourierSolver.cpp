#include "FourierSolver.h"
#include <fstream>
#define _USE_MATH_DEFINES
#define MAX_THREADS 1000
#include <math.h>
#include <iostream>
#include <QColor>

//----------------------------------------------------------------------------------------------------------------------
FourierSolver::FourierSolver() : m_width(200),m_height(200),m_rangeSelection(2.f),m_axisRange(5.f)
{
    setStandardDeviation(0.2f);
    m_psImage = QImage(m_width,m_height,QImage::Format_RGB32);
    m_pdf = new float*[m_width];
    for(int i=0;i<m_width;i++)
    {
        m_pdf[i] = new float[m_height];
        for(int j=0;j<m_height;j++)
        {
            m_pdf[i][j]=0.f;
        }
    }
    m_ps = new float*[m_width];
    for(int i=0;i<m_width;i++)
    {
        m_ps[i] = new float[m_height];
    }
}
//----------------------------------------------------------------------------------------------------------------------
FourierSolver::~FourierSolver()
{
    // Delete our histogram
    for(int i=0;i<m_width;i++)
    {
        delete [] m_pdf[i];
    }
    delete [] m_pdf;
    // Delete our power spectrum information
    for(int i=0;i<m_width;i++)
    {
        delete [] m_ps[i];
    }
    delete [] m_ps;
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
    if(file.is_open())
    {
        while (!file.eof()) {
            file >> p.x;
            file >> p.y;
            m_2DPoints.push_back(p);
        }
        file.close();
        std::cout<<"Total of "<<m_2DPoints.size()<<" points"<<std::endl;
    }
    else
    {
        std::cerr<<"Could not open file :("<<std::endl;
    }
    // Calculate our pair-wise differencials. This is a generalisation of the fourier transform to improve performance.
    // We will also create our probability density histogram here
    for(unsigned int i=0; i<m_2DPoints.size();i++)
    for(unsigned int j=0;j<m_2DPoints.size();j++)
    {
        if(i==j) continue;
        p = m_2DPoints[i]-m_2DPoints[j];
        m_differentials.push_back(p);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::importDifferentialsFromFile(QString _dir)
{
    m_differentials.clear();
    // Read our 2D points from our file into our array
    std::ifstream file(_dir.toStdString());
    float2 p;
    if(file.is_open())
    {
        while (!file.eof()) {
            file >> p.x;
            file >> p.y;
            m_differentials.push_back(p);
        }
        file.close();
        std::cout<<"Total of "<<m_differentials.size()<<" differentials"<<std::endl;
    }
    else
    {
        std::cerr<<"Could not open file :("<<std::endl;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::analysePoints()
{

    // Calculate our pair-wise differencials within our range selection
    float l;
    m_sampleDiff.clear();
    float2 p;
    for(unsigned int i=0; i<m_differentials.size();i++)
    {
        p = m_differentials[i];
        l = p.length();
        if(l>m_rangeSelection) continue;
        if(fabs(p.x)>m_axisRange||fabs(p.y)>m_axisRange) continue;

        // Add to our differentials list
        m_sampleDiff.push_back(p);

        // Build up our pfd histogram
        p+=m_axisRange;
        p/=m_axisRange+m_axisRange;
        p*=float2(m_width-1,m_height-1);
        m_pdf[(int)floor(p.x)][(int)floor(p.y)]+=1.f;
    }

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

#else
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
float FourierSolver::pdf(float2 _x)
{
    _x+=m_axisRange;
    _x/=m_axisRange+m_axisRange;
    _x*=float2(m_width-1,m_height-1);
    return (m_pdf[(int)floor(_x.x)][(int)floor(_x.y)]/(float)m_sampleDiff.size());
}
//----------------------------------------------------------------------------------------------------------------------
