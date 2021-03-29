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


#ifndef INVERSE_QUADRATIC_KERNEL_INCLUDED
#define INVERSE_QUADRATIC_KERNEL_INCLUDED

#include <vector>
#include <map>
#include <Eigen/Dense>


#include "kernel_function.hpp"

template <int d>
class InverseQuadraticKernel : public KernelFunction<d>
{
    public:
    InverseQuadraticKernel(double radius, double scale) : m_radius(radius), m_scale(scale) {};
    ~InverseQuadraticKernel() = default;
    KernelFunction<d>* new_clone() const;
    void displayParameters() const;
    int getNbParameters() const {return 1;};
    void getParameters(std::vector<double>& param) const;
    void setParameters(const double* param);
    bool is_sparse() const {return false;};
    Eigen::Matrix<double,d,d> operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;
    void get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;

    protected:
    double m_radius;
    double m_scale;
};


template <int d>
void InverseQuadraticKernel<d>::displayParameters() const
{
    std::cout << "[Radius: " << this->m_radius << " ; Scale: " << this->m_scale << std::endl;
}

template <int d>
KernelFunction<d>* InverseQuadraticKernel<d>::new_clone() const
{
    return new InverseQuadraticKernel<d>(this->m_radius,this->m_scale);
}

template <int d>
void InverseQuadraticKernel<d>::getParameters(std::vector<double>& param) const
{
    param.resize(this->getNbParameters());
  //  param[1] = m_radius;
    param[0] = m_scale;
}

template <int d>
void InverseQuadraticKernel<d>::setParameters(const double* param)
{
 //   m_radius = param[1];
    m_scale = param[0];
}


// Parametrisation with inverse of the support
template <int d>
Eigen::Matrix<double,d,d> InverseQuadraticKernel<d>::operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    Eigen::Matrix<double,d,d> I = Eigen::Matrix<double,d,d>::Identity();
    Eigen::Matrix<double,d,1> diff = x - y;
    double r = diff.norm();
    double r_scaled = r/(this->m_radius);
    double k_val = 1.0/(1.0 + std::pow<double>(r_scaled,2));        
    k_val = k_val*(this->m_scale);
    return k_val*I;
}


template <int d>
void InverseQuadraticKernel<d>::get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{ 
    partial_derivatives.resize(this->getNbParameters());
    Eigen::Matrix<double,d,d> I = Eigen::Matrix<double,d,d>::Identity();
    Eigen::Matrix<double,d,1> diff = x - y;
    double r = diff.norm();
    double r_scaled = r/(this->m_radius);
    double k_val = 1.0/(1.0 + std::pow<double>(r_scaled,2));        
    partial_derivatives[0] = k_val*I;
    val = (this->m_scale)*partial_derivatives[0];
}





#endif // REGISTRATION_AVERAGING_INCLUDED
