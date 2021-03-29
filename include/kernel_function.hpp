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


#ifndef KERNEL_FUNCTIONS_INCLUDED
#define KERNEL_FUNCTIONS_INCLUDED

#include <vector>
#include <fstream>
#include <Eigen/Dense>

template <int d>
class KernelFunction
{
    public:
    virtual void displayParameters() const = 0;
    void writeParameters(std::ofstream& infile) const;
    virtual ~KernelFunction() = 0;
    virtual int getNbParameters() const = 0;
    virtual void getParameters(std::vector<double>& param) const = 0;
    virtual void setParameters(const std::vector<double>& param);
    virtual void setParameter(int ind_parameter, double parameter_value);
    virtual void setParameters(const double* param) = 0;
    virtual bool is_sparse() const = 0;
    virtual KernelFunction<d>* new_clone() const = 0;
    virtual Eigen::Matrix<double,d,d> operator()(const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const = 0;
    virtual void get_value_and_partial_derivatives(Eigen::Matrix<double,d,d>& val, std::vector<Eigen::Matrix<double,d,d>>& partial_derivatives, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& y) const = 0;

 //   void estimateKernelHyperparametersGPP(const std::vector<Gaussian_Process_2D_Observation>& known_values, const Function_2D& mean_function);
 //   void estimateKernelHyperparametersLOO(const std::vector<Gaussian_Process_Observation<d>>& known_values, const MeanFunction<d>& mean_function);

};


template <int d>
KernelFunction<d>::~KernelFunction()
{}

template <int d>
void KernelFunction<d>::writeParameters(std::ofstream& infile) const
{
    std::vector<double> param;
    this->getParameters(param);
    for (int k=0; k<param.size(); ++k)
        infile << param[k] << " ";
}

template <int d>
void KernelFunction<d>::setParameters(const std::vector<double>& param)
{
    this->setParameters(&(param[0]));
}

template <int d>
void KernelFunction<d>::setParameter(int ind_parameter, double parameter_value)
{
    std::vector<double> current_parameters;
    this->getParameters(current_parameters);
    current_parameters[ind_parameter] = parameter_value;
    this->setParameters(current_parameters);
}


/*

template <int d>
void KernelFunction<d>::estimateKernelHyperparametersLOO(const std::vector<Gaussian_Process_Observation<d>>& known_values, const MeanFunction<d>& mean_function)
{
     // Prepare helper
    Helper_GPP_Estimation<d> helper;
    int nb_observations = known_values.size();
    int nb_parameters = this->getNbParameters();
    std::vector<Gaussian_Process_Observation<d>> known_values_copy = known_values;
    helper.known_values_ptr = &known_values_copy;
    Eigen::MatrixXd precomputed_K_XX(d*nb_observations,d*nb_observations);
    helper.precomputed_K_XX_ptr = &precomputed_K_XX;
    Eigen::MatrixXd precomputed_K_XX_inv(d*nb_observations,d*nb_observations);
    helper.precomputed_K_XX_inv_ptr = &precomputed_K_XX_inv;
    Eigen::VectorXd Y_minus_mu_X(d*nb_observations);
    fill_Y_minus_mu_X(Y_minus_mu_X,known_values,mean_function);
    helper.precomputed_Y_minus_mu_X_ptr = &Y_minus_mu_X;
    std::vector<Eigen::MatrixXd> precomputed_K_XX_deriv(nb_parameters);
    for (int p=0; p<nb_parameters; ++p)
        precomputed_K_XX_deriv[p] = Eigen::MatrixXd(d*nb_observations,d*nb_observations);
    helper.precomputed_K_XX_deriv_ptr = &precomputed_K_XX_deriv;
    helper.kernel_function_ptr = this;
    
    void *helper_ptr = &helper;
    std::vector<double> current_parameters;
    this->getParameters(current_parameters);
    for (int p=0; p<nb_parameters; ++p)
        current_parameters[p] = std::log(current_parameters[p]);

    alglib::real_1d_array x;
    x.setcontent(nb_parameters,&(current_parameters[0]));
//        double min_var(0.01), max_var(1000000);
//        bnd_l.setcontent(1,&min_var);
//        bnd_u.setcontent(1,&max_var);

    double epsg = 0.0001;
    double epsf = 0.01;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
//     alglib::minlbfgsstate state;
//     alglib::minlbfgsreport rep;
//     alglib::minlbfgscreate(1, x, state);
//    // alglib::minlbfgscreatef(nb_parameters,std::min<int>(nb_parameters,5), x, diffstep, state);
//     //  alglib::minlbfgscreatef(1,std::min<int>(nb_parameters,5), x, diffstep, state);
//  //   alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//     alglib::minlbfgsoptimize(state,computeLossFunctionHyperparameterLOOEstimation,NULL,helper_ptr);
//     alglib::minlbfgsresults(state, x, rep);

//     alglib::minlbfgsstate state;
//     alglib::minlbfgsreport rep;
//     alglib::minlbfgscreate(1, x, state);
//     alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//  //   alglib::minlbfgsoptimize(state,computeLossFunctionHyperparameterLOOEstimation,NULL,helper_ptr);
//     alglib::minlbfgsoptimize(state,computeLossFunctionHyperparameterMLEstimation,NULL,helper_ptr);
//  //   alglib::minlbfgsoptimize(state,computeLossFunctionHyperparameterGPPNoisyEstimation,NULL,helper_ptr);
//  //   alglib::minlbfgsoptimize(state,computeLossFunctionHyperparameterSMLEstimation,NULL,helper_ptr);
//     alglib::minlbfgsresults(state, x, rep);
    
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
    this->setParameters(current_parameters); 
}

*/



#endif // REGISTRATION_AVERAGING_INCLUDED
