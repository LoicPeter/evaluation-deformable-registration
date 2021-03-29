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


#ifndef KERNEL_HYPERPARAMETER_ESTIMATION_INCLUDED
#define KERNEL_HYPERPARAMETER_ESTIMATION_INCLUDED

#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
//#include <Eigen/SparseCholesky>
#include <algorithm>

// Alglib
#include "optimization.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>


#include "kernel_function.hpp"
#include "mean_function.hpp"
#include "gp_observation_database.hpp"

template <int d>
struct Helper_GPP_Estimation
{
    std::vector<std::vector<Gaussian_Process_Observation<d>>> *known_values_ptr;
    std::vector<Eigen::MatrixXd> *precomputed_K_XX_ptr;
    std::vector<std::vector<Eigen::MatrixXd>> *precomputed_K_XX_deriv_ptr;
    std::vector<Eigen::MatrixXd> *precomputed_K_XX_inv_ptr;
    std::vector<Eigen::VectorXd> *precomputed_Y_minus_mu_X_ptr;
    KernelFunction<d> *kernel_function_ptr;
    int parameter_being_optimised;
};


template <int d>
void fill_Y_minus_mu_X(Eigen::VectorXd& Y_minus_mu_X, const std::vector<Gaussian_Process_Observation<d>>& observations, const MeanFunction<d>& mean_function)
{
    int nb_observations = observations.size();
    for (int k=0; k<nb_observations; ++k)
    {
        Eigen::Matrix<double,d,1> current_point = observations[k].output_point - mean_function.operator()(observations[k].input_point);
        for (int i=0; i<d; ++i)
            Y_minus_mu_X(d*k+i) =  current_point(i);
    }
}


// K_XX_deriv[p] is the derivative of K_XX with respect to the p-th kernel hyperparameter
template <int d>
void fill_K_XX_and_derivatives(Eigen::MatrixXd& K_XX, std::vector<Eigen::MatrixXd>& K_XX_deriv, const std::vector<Gaussian_Process_Observation<d>>& observations, const KernelFunction<d>& kernel)
{
    int nb_observations = observations.size();
    int nb_parameters = K_XX_deriv.size();
    Eigen::Matrix<double,d,d> temp_block_val;
    std::vector<Eigen::Matrix<double,d,d>> temp_blocks_deriv;
    for (int k=0; k<nb_observations; ++k)
    {
        for (int l=k; l<nb_observations; ++l)
        {
            kernel.get_value_and_partial_derivatives(temp_block_val,temp_blocks_deriv,observations[k].input_point,observations[l].input_point);
            if (k==l)
                temp_block_val = temp_block_val + observations[k].observation_noise_covariance_matrix;
            K_XX.block<d,d>(k*d,l*d) = temp_block_val;
            for (int p=0; p<nb_parameters; ++p)
                K_XX_deriv[p].block<d,d>(k*d,l*d) = temp_blocks_deriv[p];
            if (k<l)
            {
                 K_XX.block<d,d>(l*d,k*d) = temp_block_val.transpose();
                 for (int p=0; p<nb_parameters; ++p)
                    K_XX_deriv[p].block<d,d>(l*d,k*d) = temp_blocks_deriv[p].transpose();
            }
        }
    }
}



template <int d>
void computeLossFunctionHyperparameterGPPNoisyEstimation(const alglib::real_1d_array& x, double &func, alglib::real_1d_array& grad, void *ptr)
{

  //  bool take_noise_into_account = true;    
    bool add_trace_term = true;

    bool check_gradients = false;
//    bool check_gradients = (grad[0]!=(-1)); // uncomment to check numerically the gradients (for example if a new kernel function is introduced)
    
    Helper_GPP_Estimation<d> *helper = static_cast<Helper_GPP_Estimation<d>*>(ptr);
    
    KernelFunction<d> *copy_kernel_function = helper->kernel_function_ptr->new_clone();
       
    int nb_parameters = copy_kernel_function->getNbParameters();
    for (int p=0; p<nb_parameters; ++p)
        copy_kernel_function->setParameter(p,std::exp(x[p])); 
    
    
    func = 0;
    for (int p=0; p<nb_parameters; ++p)
        grad[p] = 0;
    
    
    int nb_pairs = helper->known_values_ptr->size();
    
    for (int ind_pair = 0; ind_pair<nb_pairs; ++ind_pair) // loop over all the known pairs
    {
    
        double func_pair = 0;
        std::vector<double> grad_pair(nb_parameters,0);
        
        // Build K_XX + noise
        fill_K_XX_and_derivatives(helper->precomputed_K_XX_ptr->operator[](ind_pair),helper->precomputed_K_XX_deriv_ptr->operator[](ind_pair),helper->known_values_ptr->operator[](ind_pair),*copy_kernel_function);
    
        // Inverse K_XX + noise
        helper->precomputed_K_XX_inv_ptr->operator[](ind_pair) = helper->precomputed_K_XX_ptr->operator[](ind_pair).inverse();
    
        // q (could be preallocated, or better: use the function solve to not have to compute the inverse?)
        Eigen::VectorXd q = (helper->precomputed_K_XX_inv_ptr->operator[](ind_pair))*(helper->precomputed_Y_minus_mu_X_ptr->operator[](ind_pair));
    
    
        // Compute loss and gradient
        int nb_observations = helper->known_values_ptr->operator[](ind_pair).size();
        Eigen::VectorXd q_i, D_ii_inv_q_i, left_grad_vector, right_grad_vector;
        Eigen::MatrixXd D_ii, D_ii_inv, R_i, C_i, trace_part, left_trace_factor, right_trace_factor;
        for (int i=0; i<nb_observations; ++i)
        {
            // Loss function
            q_i = q.segment(d*i,d);
            D_ii = helper->precomputed_K_XX_inv_ptr->operator[](ind_pair).block(i*d,i*d,d,d);
            D_ii_inv = D_ii.inverse();
            D_ii_inv_q_i = D_ii_inv*q_i;
            double det_D_ii = D_ii.determinant();
            func_pair += d +  q_i.dot(D_ii_inv_q_i);
            if (add_trace_term)
                func_pair -= std::log(det_D_ii);
        
            // Gradient
            R_i = helper->precomputed_K_XX_inv_ptr->operator[](ind_pair).block(i*d,0,d,nb_observations*d);
            C_i = helper->precomputed_K_XX_inv_ptr->operator[](ind_pair).block(0,i*d,nb_observations*d,d);
            left_trace_factor = D_ii_inv*R_i;
            left_grad_vector = C_i*D_ii_inv_q_i;
            right_grad_vector = left_grad_vector - 2*q;
        
            for (int p=0; p<nb_parameters; ++p)
            {
                // First term
                double delta_grad = left_grad_vector.dot((helper->precomputed_K_XX_deriv_ptr->operator[](ind_pair).at(p))*right_grad_vector);
                grad_pair[p] += delta_grad;
            
                // Second term (= trace part)
                if (add_trace_term)
                {
                    trace_part = left_trace_factor*(helper->precomputed_K_XX_deriv_ptr->operator[](ind_pair).at(p))*C_i;
                    grad_pair[p] += trace_part.trace();
                }
            }
        }
        
        func_pair = 0.5*func_pair/((double)nb_observations);
        for (int p=0; p<nb_parameters; ++p)
            grad_pair[p] = 0.5*std::exp(x[p])*grad_pair[p]/((double)nb_observations);
        
        func += func_pair;
        for (int p=0; p<nb_parameters; ++p)
            grad[p] += grad_pair[p];
    }
    
    func = func/((double)nb_pairs);
    for (int p=0; p<nb_parameters; ++p)
        grad[p] = grad[p]/((double)nb_pairs);
        
    std::cout << "Current log parameters: " << x[0] << " " << x[1] << " " << x[2] << " - Loss: " << func << std::endl;
//    std::cout << "Current LOO loss: " << func << std::endl;
//    std::cout << "Current gradient: " << grad[0] << " " << grad[1] << " " << grad[2] << std::endl;
    
    delete copy_kernel_function;
    
    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.0001;
        for (int p=0; p<nb_parameters; ++p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(nb_parameters,&(x[0]));
            dummy_grad.setcontent(nb_parameters,&flag_grad);
            x_eps[p] = x[p] + eps;
            computeLossFunctionHyperparameterGPPNoisyEstimation<d>(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }
    
}


template <int d>
void estimateKernelHyperparametersLOO(KernelFunction<d>& kernel_function, const std::vector<std::vector<Gaussian_Process_Observation<d>>>& known_values, const MeanFunction<d>& mean_function)
{
      // Prepare helper
    Helper_GPP_Estimation<d> helper;
    int nb_pairs = known_values.size();
    int nb_parameters = kernel_function.getNbParameters();
    std::vector<std::vector<Gaussian_Process_Observation<d>>> known_values_copy = known_values;
    helper.known_values_ptr = &known_values_copy;
    
    std::vector<Eigen::MatrixXd> precomputed_K_XX_vec(nb_pairs), precomputed_K_XX_inv_vec(nb_pairs);
    std::vector<Eigen::VectorXd> Y_minus_mu_X_vec(nb_pairs);
    std::vector<std::vector<Eigen::MatrixXd>> precomputed_K_XX_deriv_vec(nb_pairs);
    
    
    for (int ind_pair=0; ind_pair<nb_pairs; ++ind_pair)
    {
        int nb_observations = known_values[ind_pair].size();
        precomputed_K_XX_vec[ind_pair] = Eigen::MatrixXd(d*nb_observations,d*nb_observations);
        precomputed_K_XX_inv_vec[ind_pair] = Eigen::MatrixXd(d*nb_observations,d*nb_observations);
        Y_minus_mu_X_vec[ind_pair] = Eigen::VectorXd(d*nb_observations);
        
        // Fit the mean function first
        std::shared_ptr<MeanFunction<d>> ls_transform_pair(mean_function.new_clone());
        ls_transform_pair->fit(known_values[ind_pair]);
        fill_Y_minus_mu_X(Y_minus_mu_X_vec[ind_pair],known_values[ind_pair],*ls_transform_pair);

        precomputed_K_XX_deriv_vec[ind_pair] = std::vector<Eigen::MatrixXd>(nb_parameters);
        for (int p=0; p<nb_parameters; ++p)
            precomputed_K_XX_deriv_vec[ind_pair][p] = Eigen::MatrixXd(d*nb_observations,d*nb_observations);
        
        std::cout << "Pair " << ind_pair << " - Observations: " << nb_observations << std::endl;
    }
    
   
    helper.precomputed_K_XX_ptr = &precomputed_K_XX_vec;
    helper.precomputed_K_XX_inv_ptr = &precomputed_K_XX_inv_vec;
    helper.precomputed_Y_minus_mu_X_ptr = &Y_minus_mu_X_vec;
    helper.precomputed_K_XX_deriv_ptr = &precomputed_K_XX_deriv_vec;
        
          
      
    helper.kernel_function_ptr = &kernel_function;
    
    void *helper_ptr = &helper;
    std::vector<double> current_parameters;
    kernel_function.getParameters(current_parameters);
    for (int p=0; p<nb_parameters; ++p)
        current_parameters[p] = std::log(current_parameters[p]);

    alglib::real_1d_array x;
    x.setcontent(nb_parameters,&(current_parameters[0]));
    
    //double epsg = 0.0001;
    //double epsf = 0.01;
    double epsg = 0.0000001;
    double epsf = 0.0000001;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;

    alglib::mincgstate state;
    alglib::mincgreport rep;
    alglib::mincgcreate(x,state);
    alglib::mincgsetcond(state, epsg, epsf, epsx, maxits);
  //  alglib::mincgoptimize(state,computeLossFunctionHyperparameterMLEstimation,NULL,helper_ptr);
    alglib::mincgoptimize(state,computeLossFunctionHyperparameterGPPNoisyEstimation<d>,NULL,helper_ptr);
    alglib::mincgresults(state, x, rep);

    std::cout << "New parameters: "; 
    for (int k=0; k<nb_parameters; ++k)
    {
        current_parameters[k] = std::exp(x[k]);
      //  current_parameters[k] = x[k];
        std::cout << current_parameters[k] << " ";
    }
    kernel_function.setParameters(current_parameters); 

}

template <int d>
void estimateKernelHyperparametersLOO(KernelFunction<d>& kernel_function, const std::vector<Gaussian_Process_Observation<d>>& known_values, const MeanFunction<d>& mean_function)
{
    std::vector<std::vector<Gaussian_Process_Observation<d>>> known_values_vec(1);
    known_values_vec[0] = known_values;
    estimateKernelHyperparametersLOO(kernel_function, known_values_vec, mean_function);
}

#endif // REGISTRATION_AVERAGING_INCLUDED
