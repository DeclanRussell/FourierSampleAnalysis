#include "FourierSolver.h"
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <QColor>

//----------------------------------------------------------------------------------------------------------------------
FourierSolver::FourierSolver() : m_width(100),m_height(100),m_rangeSelection(0.05),m_axisRange(0.2)
{
    m_2DPoints.clear();
    setStandardDeviation(0.05);
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
    float l;
    for(unsigned int i=0; i<m_2DPoints.size();i++)
    for(unsigned int j=0;j<m_2DPoints.size();j++)
    {
        if(i==j) continue;
        p = m_2DPoints[i]-m_2DPoints[j];
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
}
//----------------------------------------------------------------------------------------------------------------------
void FourierSolver::analysePoints()
{
    float ps;
    float2 freqVector;
    float max = 0;
    for(int x=0;x<m_psImage.width();x++)
    for(int y=0;y<m_psImage.height();y++)
    {
        freqVector = float2((float)(x-m_psImage.width()*.5f)/m_psImage.width(),(float)(y-m_psImage.height()*.5f)/m_psImage.height());
        freqVector *= m_axisRange;
        ps = m_pdf[(int)floor(x)][(int)floor(y)];//0.f;
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

}
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
