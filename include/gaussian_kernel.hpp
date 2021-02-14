#ifndef GAUSSIAN_KERNEL_INCLUDED
#define GAUSSIAN_KERNEL_INCLUDED

#include <vector>

#include <Eigen/Dense>

#include "kernel_function.hpp"

template <int d>
class GaussianKernel : public KernelFunction<d>
{
    public:
    GaussianKernel(double var, double scale, double constant_term = 0);
    ~GaussianKernel();
    KernelFunction<d>* new_clone() const;
    void displayParameters() const;
    int getNbParameters() const {return 1;};
    void getParameters(std::vector<double>& param) const;
    void setParameters(const double* param);
    bool is_sparse() const {return false;};
    Eigen::Matrix<double,d,d> operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;
    void get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;

    private:
    double m_var;
    double m_scale;
    double m_constant_term;
};



template <int d>
GaussianKernel<d>::GaussianKernel(double var, double scale, double constant_term)
{
    m_var = var;
    m_scale = scale;
    m_constant_term = constant_term;
}

template <int d>
void GaussianKernel<d>::displayParameters() const
{
    std::cout << "[Variance: " << this->m_var << " ; Scale: " << this->m_scale << " ; Constant: " << this->m_constant_term << "]" << std::endl;
}

template <int d>
KernelFunction<d>* GaussianKernel<d>::new_clone() const
{
    return new GaussianKernel<d>(this->m_var,this->m_scale,this->m_constant_term);
}

template <int d>
void GaussianKernel<d>::getParameters(std::vector<double>& param) const
{
    param.resize(1);
    //param[1] = m_var;
    param[0] = m_scale;
    //param[2] = m_constant_term;
}

template <int d>
void GaussianKernel<d>::setParameters(const double* param)
{
    //m_var = param[1];
    m_scale = param[0];
    //m_constant_term = param[2];
}


template <int d>
GaussianKernel<d>::~GaussianKernel()
{}

template <int d>
Eigen::Matrix<double,d,d> GaussianKernel<d>::operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    Eigen::Matrix<double,d,1> diff = x - y;
    double k_val = (this->m_scale)*std::exp(-(diff.squaredNorm())/(2*(this->m_var)));
    if (x==y)
        k_val += this->m_constant_term;
    return k_val*(Eigen::Matrix<double,d,d>::Identity());
}

template <int d>
void GaussianKernel<d>::get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    partial_derivatives.resize(1);
    
    Eigen::Matrix<double,d,d> I = Eigen::Matrix<double,d,d>::Identity();
    Eigen::Matrix<double,d,1> diff = x - y;
    
    double r_2 = diff.squaredNorm();
    double exp_fact = std::exp(-r_2/(2*(this->m_var)));
    partial_derivatives[0] = exp_fact*I;
    double k_val = (this->m_scale)*exp_fact;
    double deriv_0 = (k_val*r_2)/(2*std::pow<double>(this->m_var,2));
   // partial_derivatives[1] = deriv_0*I;
    if (x==y)
    {
        k_val += this->m_constant_term;
        //partial_derivatives[2] = I;
    }
    //else
    //    partial_derivatives[2] = Eigen::Matrix<double,d,d>::Zero();
    val = k_val*I;
}




#endif // REGISTRATION_AVERAGING_INCLUDED
