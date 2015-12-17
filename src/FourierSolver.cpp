#include "FourierSolver.h"
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <QColor>

//----------------------------------------------------------------------------------------------------------------------
FourierSolver::FourierSolver() : m_width(1000),m_height(1000)
{
    m_2DPoints.clear();
    setStandardDeviation(0.5);
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
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::import2DFromFile(QString _dir)
{
    // Delete any data we may have already imported
    m_2DPoints.clear();
    // Read our 2D points from our file into our array
    std::ifstream file(_dir.toStdString());
    float2 p;
    int idx=0;
    if(file.is_open())
    {
        while (!file.eof()) {
            file >> p.x;
            file >> p.y;
            m_2DPoints.push_back(p);
            std::cout<<"p"<<idx<<" = ("<<p.x<<","<<p.y<<")"<<std::endl;
            idx++;
        }
        file.close();
        std::cout<<"Total of "<<m_2DPoints.size()<<" points"<<std::endl;
    }
    else
    {
        std::cerr<<"Could not open file :("<<std::endl;
    }
    // first find our min and max values
    float2 min,max;
    min = max = m_2DPoints[0];
    for(unsigned int i=0; i<m_2DPoints.size();i++)
    {
        if(m_2DPoints[i].x<min.x) min.x = m_2DPoints[i].x;
        if(m_2DPoints[i].y<min.y) min.y = m_2DPoints[i].y;
        if(m_2DPoints[i].x>max.x) max.x = m_2DPoints[i].x;
        if(m_2DPoints[i].y>max.y) max.y = m_2DPoints[i].y;
    }
    // Move our points between 0-1
    max-=min;
    for(unsigned int i=0; i<m_2DPoints.size();i++)
    {
        m_2DPoints[i]-=min;
        m_2DPoints[i]/=max;
    }
    // Calculate our pair-wise differencials. This is a generalisation of the fourier transform to improve performance.
    // We will also create our probability density histogram here
    float pd = 1.f/((m_2DPoints.size()-1) * (m_2DPoints.size()-1));
    for(unsigned int i=0; i<m_2DPoints.size();i++)
    for(unsigned int j=0;j<m_2DPoints.size();j++)
    {
        if(i==j) continue;
        p = m_2DPoints[i]-m_2DPoints[j];
        if(p.length()<0.01)
        {
            m_sampleDiff.push_back(p);
        }
//        m_diffLength.push_back(p.length());
        p+=float2(1,1);
        p/=float2(2,2);
        p*=float2(m_width-1,m_height-1);
        m_pdf[(int)floor(p.x)][(int)floor(p.y)]+= pd;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::analysePoints()
{
    float ps;
    float2 freqVector;
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
//        freqVector = float2((float)(x-m_psImage.width()*.5f)/m_psImage.width(),(float)(y-m_psImage.height()*.5f)/m_psImage.height());
//        ps = 0.f;
//        for(unsigned int i=0;i<m_sampleDiff.size();i++)
//        {
//            //if(m_diffLength[i]>0.1) continue;
//            ps+= gausian(freqVector,m_sampleDiff[i])*pdf(m_sampleDiff[i]);
//        }
        //ps*=m_2DPoints.size();
        ps = m_pdf[x][y]*25500000;
        std::cout<<"ps" <<ps<<std::endl;
        m_psImage.setPixel(x,y,QColor(ps,ps,ps).rgb());
    }
}
//----------------------------------------------------------------------------------------------------------------------
float FourierSolver::gausian(float2 _q, float2 _d)
{
    float2 sq = (_q-_d);
    sq*=sq;
    float w = pow(M_E,-((sq.x/m_stanDevSqrd)+(sq.y/m_stanDevSqrd)));
    return w;
}
//----------------------------------------------------------------------------------------------------------------------
float FourierSolver::pdf(float2 _x)
{
    _x+=float2(1,1);
    _x/=float2(2,2);
    _x*=float2(m_width-1,m_height-1);
    return m_pdf[(int)floor(_x.x)][(int)floor(_x.y)];
}
//----------------------------------------------------------------------------------------------------------------------
