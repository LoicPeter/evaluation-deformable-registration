#ifndef KERNEL_BUNDLE_INCLUDED
#define KERNEL_BUNDLE_INCLUDED

#include <vector>
#include <Eigen/Dense>
#include "kernel_function.hpp"
#include "gaussian_kernel.hpp"
#include "wendland_kernel.hpp"
#include "constant_kernel.hpp"
#include "inverse_quadratic_kernel.hpp"

template <int d>
class KernelBundle : public KernelFunction<d>
{
    public:
    KernelBundle();
    ~KernelBundle();
    void displayParameters() const;
    void add_to_bundle(const KernelFunction<d>& kernel);
    KernelFunction<d>* new_clone() const;
    int getNbParameters() const;
    void getParameters(std::vector<double>& param) const;
    void setParameters(const double* param);
    bool is_sparse() const;
    Eigen::Matrix<double,d,d> operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;
    void get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const;
    
    protected:
    std::vector<KernelFunction<d>*> m_kernels;
};



template <int d>
KernelBundle<d>::KernelBundle()
{}

template <int d>
KernelBundle<d>::~KernelBundle()
{
    int nb_kernels = m_kernels.size();
    for (int k=0; k<nb_kernels; ++k)
        delete m_kernels[k];
}

template <int d>
void KernelBundle<d>::add_to_bundle(const KernelFunction<d>& kernel)
{
    KernelFunction<d>* new_kernel = kernel.new_clone();
    m_kernels.push_back(new_kernel);
}

template <int d>
KernelFunction<d>* KernelBundle<d>::new_clone() const
{
    KernelBundle<d> *p = new KernelBundle<d>();
    int nb_kernels = m_kernels.size();
    for (int k=0; k<nb_kernels; ++k)
        p->add_to_bundle(*m_kernels[k]);
    return p;
}

template <int d>
void KernelBundle<d>::displayParameters() const
{
    int nb_kernels = m_kernels.size();
    for (int k=0; k<nb_kernels; ++k)
    {
        std::cout << "Parameters kernel #" << k << ": ";
        m_kernels[k]->displayParameters();
        std::cout << std::endl;
    }
}

template <int d>
int KernelBundle<d>::getNbParameters() const
{
    int res(0);
    int nb_kernels = m_kernels.size();
    for (int k=0; k<nb_kernels; ++k)
        res += m_kernels[k]->getNbParameters();
    return res;
}

template <int d>
void KernelBundle<d>::getParameters(std::vector<double>& param) const
{
    int nb_parameters = this->getNbParameters();
    param.resize(nb_parameters);
    int nb_kernels = m_kernels.size();
    std::vector<double> kernel_param;
    int i(0);
    for (int k=0; k<nb_kernels; ++k)
    {
        m_kernels[k]->getParameters(kernel_param);
        for (int j=0; j<kernel_param.size(); ++j)
        {
            param[i] = kernel_param[j];
            ++i;
        }
    }
}

template <int d>
void KernelBundle<d>::setParameters(const double* param)
{
    int nb_parameters = this->getNbParameters();
    int nb_kernels = m_kernels.size();
    int i(0);
    for (int k=0; k<nb_kernels; ++k)
    {
        m_kernels[k]->setParameters(param + i);
        i += m_kernels[k]->getNbParameters();
    }
}

template <int d>
bool KernelBundle<d>::is_sparse() const
{
    bool res(true);
    int nb_kernels = m_kernels.size();
    for (int k=0; k<nb_kernels; ++k)
    {
        if (m_kernels[k]->is_sparse()==false)
            res = false;
    }
    return res;
}

template <int d>
Eigen::Matrix<double,d,d> KernelBundle<d>::operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    Eigen::Matrix<double,d,d> res = Eigen::Matrix<double,d,d>::Zero();
    int nb_kernels = m_kernels.size();
    for (int k=0; k<nb_kernels; ++k)
        res += m_kernels[k]->operator()(x,y);
    return res;
}

template <int d>
void KernelBundle<d>::get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const
{
    int nb_kernels = m_kernels.size();
    int nb_parameters = this->getNbParameters();
    val = Eigen::Matrix<double,d,d>::Zero();
    partial_derivatives.resize(nb_parameters);
    int i(0);
    for (int k=0; k<nb_kernels; ++k)
    {
        Eigen::Matrix<double,d,d> kernel_val;
        std::vector<Eigen::Matrix<double,d,d>> kernel_partial_derivatives;
        m_kernels[k]->get_value_and_partial_derivatives(kernel_val,kernel_partial_derivatives,x,y);
        val += kernel_val;
        int nb_kernel_parameters = kernel_partial_derivatives.size();
        for (int p=0; p<nb_kernel_parameters; ++p)
        {
            partial_derivatives[i] = kernel_partial_derivatives[p];
            ++i;
        }
    }
}


// template <int d>
// void create_multiscale_wendland(KernelBundle<d>& kernel, int nb_levels, double max_radius, double scale, int min_s = 0)
// {
//     for (int wd=1; wd<=1; ++wd)
//     {
//         for (int s=min_s; s<nb_levels; ++s)
//         {
//             double inv_radius = (std::pow<double>(2,s))/max_radius;
//             WendlandKernel<d> wendland_term(inv_radius,scale,0,wd);
//             kernel.add_to_bundle(wendland_term);
//         }
//     }
//     
//   //  ConstantKernel<d> constant_term(scale);
//   //  kernel.add_to_bundle(constant_term);
// }
// 
// template <int d>
// void create_multiscale_gaussian(KernelBundle<d>& kernel, int nb_levels, double max_radius, double scale, int min_s = 0)
// {
//     for (int s=min_s; s<nb_levels; ++s)
//     {
//         double radius = max_radius/(std::pow<double>(2,s)); // this is the radius equivalent to support in Wendland kernel
//         double eq_var = (2*std::pow<double>(radius,2))/(9*M_PI); // equivalent variance
//         GaussianKernel<d> gaussian_term(eq_var,scale);
//         kernel.add_to_bundle(gaussian_term);
//     }
//     
//   //  ConstantKernel<d> constant_term(scale);
//   //  kernel.add_to_bundle(constant_term);
// }
// 
// 
// template <int d>
// void create_multiscale_inverse_quadratic(KernelBundle<d>& kernel, int nb_levels, double max_radius, double scale, int min_s = 0)
// {
//     for (int s=min_s; s<nb_levels; ++s)
//     {
//         double radius = max_radius/(std::pow<double>(2,s)); // this is the radius equivalent to support in Wendland kernel
//         double eq_radius = (2*radius)/(3*M_PI); // equivalent variance
//         InverseQuadraticKernel<d> iq_term(eq_radius,scale);
//         kernel.add_to_bundle(iq_term);
//     }
//     
//   //  ConstantKernel<d> constant_term(scale);
//   //  kernel.add_to_bundle(constant_term);
// }


template <int d>
void create_multiscale_wendland(KernelBundle<d>& kernel, int nb_levels, double min_radius, double scale, int min_s = 0)
{
    for (int wd=1; wd<=1; ++wd)
    {
        for (int s=min_s; s<nb_levels; ++s)
        {
            double radius = (std::pow<double>(2,s))*min_radius;
            double inv_radius = 1.0/radius;
            WendlandKernel<d> wendland_term(inv_radius,scale,0,wd);
            kernel.add_to_bundle(wendland_term);
        }
    }
    
  //  ConstantKernel<d> constant_term(scale);
  //  kernel.add_to_bundle(constant_term);
}

template <int d>
void create_multiscale_gaussian(KernelBundle<d>& kernel, int nb_levels, double min_radius, double scale, int min_s = 0)
{
    for (int s=min_s; s<nb_levels; ++s)
    {
        double radius = min_radius*(std::pow<double>(2,s)); // this is the radius equivalent to support in Wendland kernel
        double eq_var = (2*std::pow<double>(radius,2))/(9*M_PI); // equivalent variance
        GaussianKernel<d> gaussian_term(eq_var,scale);
        kernel.add_to_bundle(gaussian_term);
    }
    
  //  ConstantKernel<d> constant_term(scale);
  //  kernel.add_to_bundle(constant_term);
}


template <int d>
void create_multiscale_inverse_quadratic(KernelBundle<d>& kernel, int nb_levels, double min_radius, double scale, int min_s = 0)
{
    for (int s=min_s; s<nb_levels; ++s)
    {
        double radius = min_radius*(std::pow<double>(2,s)); // this is the radius equivalent to support in Wendland kernel
        double eq_radius = (2*radius)/(3*M_PI); // equivalent variance
        InverseQuadraticKernel<d> iq_term(eq_radius,scale);
        kernel.add_to_bundle(iq_term);
    }
    
  //  ConstantKernel<d> constant_term(scale);
  //  kernel.add_to_bundle(constant_term);
}



#endif // REGISTRATION_AVERAGING_INCLUDED
