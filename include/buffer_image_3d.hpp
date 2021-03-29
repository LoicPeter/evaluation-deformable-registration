// -------------------------------------------------------------------
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
// -------------------------------------------------------------------



#ifndef BUFFER_IMAGE_3D_HPP
#define BUFFER_IMAGE_3D_HPP

#include <vector>
#include <stdexcept>

template <class PixelType>
class BufferImage3D
{
public:
    BufferImage3D(int dimx, int dimy, int dimz);
    BufferImage3D(int dimx, int dimy, int dimz, PixelType initValue);
    int getDimension(int dim) const;
    inline PixelType* data() {return &(m_data[0]);};
    inline const PixelType* data() const {return &(m_data[0]);};
    inline int size() const {return (m_dimx_times_dimy*m_dimz);};
    inline PixelType& at(int x, int y, int z) {return m_data[m_dimx_times_dimy*z + m_dimx*y + x];};
    inline const PixelType& at(int x, int y, int z) const {return m_data[m_dimx_times_dimy*z + m_dimx*y + x];};
    PixelType interpolate(double x, double y, double z) const;

    
private:
	std::vector<PixelType> m_data;
	int m_dimx;
	int m_dimy;
	int m_dimz;
    int m_dimx_times_dimy;
};

template <class PixelType>
BufferImage3D<PixelType> operator*(double lambda, const BufferImage3D<PixelType>& b)
{
    BufferImage3D<PixelType> res = b;
    int N = res.size();
    PixelType *data_ptr = res.data();
    for (int k=0; k<N; ++k)
        data_ptr[k] *= lambda;
    return res;
}

template <class PixelType>
void extract_maxima(std::vector<Eigen::Vector3i>& maxima, const BufferImage3D<PixelType>& volume)
{
    maxima.clear();
    int dimx = volume.getDimension(0);
    int dimy = volume.getDimension(1);
    int dimz = volume.getDimension(2);
    for (int z=1; z<(dimz-1); ++z)
    {
        for (int y=1; y<(dimy-1); ++y)
        {
            for (int x=1; x<(dimx-1); ++x)
            {
                PixelType val = volume.at(x,y,z);
                bool isMaximum(true);
                for (int dz=-1; dz<=1; ++dz)
                {
                    for (int dy=-1; dy<=1; ++dy)
                    {
                        for (int dx=-1; dx<=1; ++dx)
                        {
                            if ((val<=volume.at(x+dx,y+dy,z+dz)) && ((dx!=0) || (dy!=0) || (dz!=0)))
                                isMaximum = false;
                        }
                    }
                }
                if (isMaximum)
                    maxima.push_back(Eigen::Vector3i(x,y,z));
            }
        }
    }
}

template <class PixelType>
void multiply_in_place(double lambda, BufferImage3D<PixelType>& b)
{
    int N = b.size();
    PixelType *data_ptr = b.data();
    for (int k=0; k<N; ++k)
        data_ptr[k] *= lambda;
}

// Output buffer is supposed to already have the proper size
template <class PixelType> 
void resize(BufferImage3D<PixelType>& output_buffer, const BufferImage3D<PixelType>& input_buffer)
{
    int dimx_small = input_buffer.getDimension(0);
    int dimy_small = input_buffer.getDimension(1);
    int dimz_small = input_buffer.getDimension(2);
    int dimx_large = output_buffer.getDimension(0);
    int dimy_large = output_buffer.getDimension(1);
    int dimz_large = output_buffer.getDimension(2);
    
    for (int z=0; z<dimz_large; ++z)
    {
        for (int y=0; y<dimy_large; ++y)
        {
            for (int x=0; x<dimx_large; ++x)
            {
                double xs = ((double)((dimx_small-1)*x))/((double)dimx_large - 1);
                double ys = ((double)((dimy_small-1)*y))/((double)dimy_large - 1);
                double zs = ((double)((dimz_small-1)*z))/((double)dimz_large - 1);
                output_buffer.at(x,y,z) = input_buffer.interpolate(xs,ys,zs);
            }    
        }
    }
}




template <class PixelType>
BufferImage3D<PixelType>::BufferImage3D(int dimx, int dimy, int dimz) : m_dimx(dimx), m_dimy(dimy), m_dimz(dimz)
{
    m_dimx_times_dimy = dimx*dimy;
    m_data.resize(this->size());
}

template <class PixelType>
BufferImage3D<PixelType>::BufferImage3D(int dimx, int dimy, int dimz, PixelType initValue) : m_dimx(dimx), m_dimy(dimy), m_dimz(dimz)
{
    m_dimx_times_dimy = dimx*dimy;
    m_data = std::vector<PixelType>(this->size(),initValue);
}


template <class PixelType>
int BufferImage3D<PixelType>::getDimension(int dim) const
{
    if (dim==0)
        return m_dimx;
    else 
    {
        if (dim==1)
            return m_dimy;
        else
        {
            if (dim==2)
                return m_dimz;
            else
                throw std::out_of_range("In BufferImage3D: a dimension was asked that is not 0, 1 or 2");
        }
    }
}


template <class PixelType>
PixelType BufferImage3D<PixelType>::interpolate(double x, double y, double z) const
{
    // Put the point back in the volume if needed
    double eps = 0.0000001;
    if (x<0)
        x = 0;
    if (x>=(m_dimx - 1))
        x = m_dimx - 1 - eps;
    if (y<0)
        y = 0;
    if (y>=(m_dimy - 1))
        y = m_dimy - 1 - eps;
    if (z<0)
        z = 0;
    if (z>=(m_dimz - 1))
        z = m_dimz - 1 - eps;
    
    // Define the cube surrounding (x,y,z)
	int x1 = std::floor(x);
	int y1 = std::floor(y);
	int z1 = std::floor(z);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    int z2 = z1 + 1;
    double dx = x - x1;
    double dy = y - y1;
    double dz = z - z1;
        
    // Values at the 8 corners of the cube
    PixelType f_111 = this->at(x1,y1,z1);
    PixelType f_112 = this->at(x1,y1,z2);
    PixelType f_121 = this->at(x1,y2,z1);
    PixelType f_122 = this->at(x1,y2,z2);
    PixelType f_211 = this->at(x2,y1,z1);
    PixelType f_212 = this->at(x2,y1,z2);
    PixelType f_221 = this->at(x2,y2,z1);
    PixelType f_222 = this->at(x2,y2,z2);
    
    // Interpolate
//     double v_x11 = dx*v_211 + (1 - dx)*v_111;
//     double v_x12 = dx*v_212 + (1 - dx)*v_112;
//     double v_x21 = dx*v_221 + (1 - dx)*v_121;
//     double v_x22 = dx*v_222 + (1 - dx)*v_122;
//     double v_xy1 = dy*v_x21 + (1 - dy)*v_x11;
//     double v_xy2 = dy*v_x22 + (1 - dy)*v_x12;
//     double v_xyz = dz*v_xy2 + (1 - dz)*v_xy1;
    
    PixelType val = dx*dy*dz*f_222 + (1-dx)*dy*dz*f_122 + dx*(1-dy)*dz*f_212 + (1-dx)*(1-dy)*dz*f_112 + dx*dy*(1-dz)*f_221 + (1-dx)*dy*(1-dz)*f_121 + dx*(1-dy)*(1-dz)*f_211 + (1-dx)*(1-dy)*(1-dz)*f_111;
        
    return val;
}



#endif
