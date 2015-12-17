#ifndef FLOAT2
#define FLOAT2

//----------------------------------------------------------------------------------------------------------------------
/// @brief simple structure to represent 2D points
//----------------------------------------------------------------------------------------------------------------------
struct float2
{
    float x;
    float y;
    float2(float _x=0.f, float _y=0.f)
    {
        x=_x;
        y=_y;
    }
    inline float2& operator =(const float2 &f)
    {
        x = f.x;
        y = f.y;
        return *this;
    }
    inline float2 operator +(const float2 &f)
    {
        return float2(x+f.x,y+f.y);
    }
    inline float2 operator -(const float2 &f)
    {
        return float2(x-f.x,y-f.y);
    }
    inline float2 operator /(const float2 &f)
    {
        return float2(x/f.x,y/f.y);
    }
    inline float2 operator *(const float2 &f)
    {
        return float2(x*f.x,y*f.y);
    }
    inline void operator +=(const float2 &f)
    {
        x+=f.x;
        y+=f.y;
    }
    inline void operator -=(const float2 &f)
    {
        x-=f.x;
        y-=f.y;
    }
    inline void operator /=(const float2 &f)
    {
        x/=f.x;
        y/=f.y;
    }
    inline void operator *=(const float2 &f)
    {
        x*=f.x;
        y*=f.y;
    }
    inline float dot(float2 &f)
    {
        return x*f.x + y*f.y;
    }
    inline float length()
    {
        return sqrt(x*x + y*y);
    }
    inline void operator +=(const float &f)
    {
        x+=f;
        y+=f;
    }
    inline void operator -=(const float &f)
    {
        x-=f;
        y-=f;
    }
    inline void operator /=(const float &f)
    {
        x/=f;
        y/=f;
    }
    inline void operator *=(const float &f)
    {
        x*=f;
        y*=f;
    }
    inline void operator +(const float &f)
    {
        x+=f;
        y+=f;
    }
    inline void operator -(const float &f)
    {
        x-=f;
        y-=f;
    }
    inline void operator /(const float &f)
    {
        x/=f;
        y/=f;
    }
    inline void operator *(const float &f)
    {
        x*=f;
        y*=f;
    }
};

#endif // FLOAT2

