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


#ifndef WENDLAND_KERNEL_INCLUDED
#define WENDLAND_KERNEL_INCLUDED

#include <vector>
#include <map>
#include <Eigen/Dense>


#include "kernel_function.hpp"

template <int d>
class WendlandKernel : public KernelFunction<d>
{
    public:
    WendlandKernel(double radius, double scale, double constant_term = 0, int k = 0);
    ~WendlandKernel();
    KernelFunction<d>* new_clone() const;
    void displayParameters() const;
    int getNbParameters() const {return 1;};
    void getParameters(std::vector<double>& param) const;
    void setParameters(const double* param);
    bool is_sparse() const {return true;};
    Eigen::Matrix<double,d,d> operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;
    void get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;

    protected:
    double m_radius;
    double m_scale;
    double m_constant_term;
    int m_k;
};




template <int d>
WendlandKernel<d>::WendlandKernel(double radius, double scale, double constant_term, int k)
{
    static_assert(((d==2) || (d==3)),"Wendland kernels are only implemented in dimension 2 or 3");
    m_radius = radius;
    m_scale = scale;
    m_constant_term = constant_term;
    m_k = k;
}

template <int d>
void WendlandKernel<d>::displayParameters() const
{
    std::cout << "[Radius: " << this->m_radius << " ; Scale: " << this->m_scale << " ; Constant: " << this->m_constant_term << "]" << std::endl;
}

template <int d>
KernelFunction<d>* WendlandKernel<d>::new_clone() const
{
    return new WendlandKernel<d>(this->m_radius,this->m_scale,this->m_constant_term,this->m_k);
}

template <int d>
void WendlandKernel<d>::getParameters(std::vector<double>& param) const
{
    param.resize(this->getNbParameters());
  //  param[1] = m_radius;
    param[0] = m_scale;
 //   param[1] = m_constant_term;
}

template <int d>
void WendlandKernel<d>::setParameters(const double* param)
{
 //   m_radius = param[1];
    m_scale = param[0];
//   m_constant_term = param[1];
}

template <int d>
WendlandKernel<d>::~WendlandKernel()
{}


// Parametrisation with inverse of the support
template <int d>
Eigen::Matrix<double,d,d> WendlandKernel<d>::operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    auto positive_part = [](double x) {return std::max<double>(0,x);};
    
    Eigen::Matrix<double,d,d> I = Eigen::Matrix<double,d,d>::Identity();
    Eigen::Matrix<double,d,1> diff = x - y;
    double r = diff.norm();
    double r_scaled = r*(this->m_radius);
    double k_val(-1);
    if (m_k==0)
        k_val = std::pow<double>(positive_part(1-r_scaled),2);
    if (m_k==1)
        k_val = (4*r_scaled + 1)*std::pow<double>(positive_part(1-r_scaled),4);
    if (m_k==2)
        k_val = (35*r_scaled*r_scaled + 18*r_scaled  + 3)*std::pow<double>(positive_part(1-r_scaled),6);
    if (k_val==(-1))
        std::cout << "The index k used for the Wendland kernel is not supported yet (k = 0,1,2 only so far)" << std::endl; 
        
    k_val = k_val*(this->m_scale);
    if (x==y)
        k_val += this->m_constant_term;
   
    return k_val*I;
}


template <int d>
void WendlandKernel<d>::get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    auto positive_part = [](double x) {return std::max<double>(0,x);};
    
    partial_derivatives.resize(this->getNbParameters());
    Eigen::Matrix<double,d,d> I = Eigen::Matrix<double,d,d>::Identity();
    Eigen::Matrix<double,d,1> diff = x - y;
    double r = diff.norm();
    double r_scaled = r*(this->m_radius);
    int ind_exp = 2*(m_k+1);
    double positive_part_fact = positive_part(1-r_scaled);
    double positive_part_fact_ind_exp = std::pow<double>(positive_part_fact,ind_exp);
    
    double fact(-1), deriv_fact(-1); // deriv_fact is the derivative of the factor wrt m_radius (first parameter)
    if (m_k==0)
    {
        fact = 1;
        deriv_fact = 0;
    }
    if (m_k==1)
    {
        fact = 4*r_scaled + 1;
        deriv_fact = 4*r;
    }
    if (m_k==2)
    {
        fact = 35*r_scaled*r_scaled + 18*r_scaled + 3;
        deriv_fact = 70*r_scaled*r + 18*r;
    }
    if (fact==(-1))
        std::cout << "The index k used for the Wendland kernel is not supported yet (k = 0,1,2 only so far)" << std::endl; 

    double k_val = fact*positive_part_fact_ind_exp;
    partial_derivatives[0] = k_val*I;
    k_val = k_val*(this->m_scale);
    
    
    // Derivative
  //  double deriv_first_parameter(0);
  //  deriv_first_parameter -= fact*ind_exp*(this->m_scale)*std::pow<double>(positive_part_fact,ind_exp-1)*r;
  //  deriv_first_parameter += (this->m_scale)*positive_part_fact_ind_exp*deriv_fact;
    
   // partial_derivatives[1] = deriv_first_parameter*I; // deriv wrt radius
    
    
    if (x==y)
    {
       // partial_derivatives[1] = I;
        k_val += this->m_constant_term;
    }
    else
    {
       // partial_derivatives[1] = Eigen::MatrixXd::Zero(2,2);
    }
    
    val = k_val*I;
    
//    std::cout << "Check empirically the derivatives" << std::endl;
//    std::cout << partial_derivatives[0] << " " << partial_derivatives[1] << " " <<  partial_derivatives[2] << std::endl;
}





#endif // REGISTRATION_AVERAGING_INCLUDED
