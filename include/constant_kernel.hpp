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


#ifndef CONSTANT_KERNEL_INCLUDED
#define CONSTANT_KERNEL_INCLUDED

#include <vector>
#include <Eigen/Dense>
#include "kernel_function.hpp"

template <int d>
class ConstantKernel : public KernelFunction<d>
{
    public:
    ConstantKernel(double scale);
    ~ConstantKernel();
    KernelFunction<d>* new_clone() const;
    void displayParameters() const;
    int getNbParameters() const {return 1;};
    void getParameters(std::vector<double>& param) const;
    void setParameters(const double* param);
    bool is_sparse() const {return false;};
    Eigen::Matrix<double,d,d> operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;
    void get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;

    protected:
    double m_scale;
    Eigen::Matrix<double,d,d> m_I; // identity matrix
};

//-----------------------------------------------
// Constant (isotropic) kernel
//-----------------------------------------------

template <int d>
ConstantKernel<d>::ConstantKernel(double scale)
{
    m_scale = scale;
    m_I = Eigen::Matrix<double,d,d>::Identity();
}

template <int d>
ConstantKernel<d>::~ConstantKernel()
{}

template <int d>
KernelFunction<d>* ConstantKernel<d>::new_clone() const
{
    ConstantKernel<d> *p = new ConstantKernel<d>(m_scale);
    return p;
}

template <int d>
void ConstantKernel<d>::displayParameters() const
{
    std::cout << "Scale: " << m_scale << std::endl;
}

template <int d>
void ConstantKernel<d>::getParameters(std::vector<double>& param) const
{
    param.resize(1);
    param[0] = m_scale;
}

template <int d>
void ConstantKernel<d>::setParameters(const double* param)
{
    m_scale = param[0];    
}   

template <int d>
Eigen::Matrix<double,d,d> ConstantKernel<d>::operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    return (m_scale*m_I);
}

template <int d>
void ConstantKernel<d>::get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    val = m_scale*m_I;
    partial_derivatives.resize(1);
    partial_derivatives[0] = m_I;
}


#endif // REGISTRATION_AVERAGING_INCLUDED
