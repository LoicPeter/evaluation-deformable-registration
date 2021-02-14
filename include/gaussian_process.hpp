#ifndef GAUSSIAN_PROCESS_INCLUDED
#define GAUSSIAN_PROCESS_INCLUDED

#include <vector>
#include <math.h>
#include <set>
#include <fstream>
#include <Eigen/Dense>
#include "kernel_function.hpp"
#include "mean_function.hpp"
#include "gp_observation_database.hpp"
#include "updatable_cholesky.hpp"


enum EvaluationMeasure {l1, l2, l_inf};

struct Helper_Entropy_Decrease_Computation
{
    //Eigen::MatrixXd K_T_T;
    Eigen::LLT<Eigen::MatrixXd> K_T_T_cholesky;
    Eigen::LLT<Eigen::MatrixXd> A_cholesky;
  //  Eigen::MatrixXd K_X_T;
 //   Eigen::MatrixXd K_X_T_tr;
    Eigen::MatrixXd K_T_T_inv_times_K_X_T_tr;
};


template <int d>
class Gaussian_Process
{
    public:
    Gaussian_Process(const MeanFunction<d>& mean_function, const KernelFunction<d>& covariance_function);
    ~Gaussian_Process();
    KernelFunction<d>* new_covariance_function() {return m_unconditioned_covariance_function->new_clone();};
    void reset(const MeanFunction<d>& mean_function, const KernelFunction<d>& covariance_function, const std::vector<Gaussian_Process_Observation<d>>& known_values);
    bool is_kernel_sparse() const;
    void writeParameters(std::ofstream& infile) const {m_unconditioned_covariance_function->writeParameters(infile);};
    void mean(Eigen::Matrix<double,d,1>& mean, const Eigen::Matrix<double,d,1>& x);
    void covariance(Eigen::Matrix<double,d,d>& covariance, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& x_prime);
    void unconditioned_covariance(Eigen::Matrix<double,d,d>& covariance, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& x_prime) {covariance = this->m_unconditioned_covariance_function->operator()(x,x_prime);};
    void predict_joint_covariance(Eigen::MatrixXd& res, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations);
    void predict_joint_covariance(Eigen::MatrixXd& res, const Eigen::MatrixXd& unconditioned_covariance, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations);
    void predict_joint_unconditioned_covariance(Eigen::MatrixXd& res, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations);
    void setKnownValues(const std::vector<Gaussian_Process_Observation<d>>& known_values);
    void addKnownValue(const Gaussian_Process_Observation<d>& obs);
 //   void setKnownValues(const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& known_values, double observation_noise_variance);
    void clearKnownValues();
    void reestimate_unconditioned_mean(const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& known_values);
    void reestimate_covariance_hyperparameters(const std::vector<Gaussian_Process_Observation<d>>& known_values);
    void set_predefined_candidate_locations(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& candidate_locations);
    void set_predefined_candidate_locations(const std::vector<std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>>& candidate_locations);
    void compute_unconditioned_K_XT(Eigen::MatrixXd& K_XT, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& T);
    
    // Entropy decrease computation
    double compute_entropy_decrease(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations, const Eigen::Matrix<double,d,1>& added_location);
    void create_arbitrary_covariance_matrix(Eigen::MatrixXd& K, const std::vector<Eigen::Matrix<double,d,1>>& input_locations_i, const std::vector<Eigen::Matrix<double,d,1>>& input_locations_j, bool use_unconditioned_covariance_function);
    void precompute_helper_for_entropy_decrease_computation(Helper_Entropy_Decrease_Computation& helper, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations);
    double compute_entropy_decrease(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations, const Eigen::Matrix<double,d,1>& added_location, const Helper_Entropy_Decrease_Computation& helper);
    double compute_entropy(const Eigen::Matrix<double,d,1>& x);
    double compute_entropy(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations);
    void compute_entropies_of_predefined_candidates(std::map<Eigen::Matrix<double,d,1>,double,comp_Point<d>>& entropies, int ind_predefined_candidate_set);
    double compute_trace_covariance_predefined_candidates(int ind_predefined_candidate_set);
    void predict_transformation_at_predefined_candidates(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& predicted_transformation, int ind_predefined_candidate_set);
    void compute_covariance_matrix_at_predefined_candidates(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>,comp_Point<d>>& covariance_matrices, int ind_predefined_candidate_set);
    
    // Evaluation
    void evaluate_quality(std::ofstream& infile, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations);
    void evaluate_quality_on_predefined_set(std::ofstream& infile, std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& predicted_transformation, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, int ind_predefined_candidate_set);
    void evaluate_transformations_on_predefined_set(std::vector<std::vector<double>>& transformation_scores, const std::vector<std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>>& transformations_to_evaluate, int ind_predefined_candidate_set, EvaluationMeasure eval_measure);
    
    private:
    MeanFunction<d>* m_unconditioned_mean_function;
    KernelFunction<d>* m_unconditioned_covariance_function;
    
    // For conditoning
    std::vector<Gaussian_Process_Observation<d>> m_known_values;
    Eigen::MatrixXd K_XX;
    Eigen::MatrixXd observation_noise_matrix;
    Eigen::MatrixXd K_XX_plus_noise_inv;
    Eigen::VectorXd Y_minus_mu_X;
    Eigen::MatrixXd precomputed_matrix_product_for_mean;
    
    Eigen::MatrixXd preallocated_matrix_for_K_X_tr;
    Eigen::MatrixXd preallocated_matrix_for_K_X_prime;
  //  Eigen::LLT<Eigen::MatrixXd> K_XX_plus_noise_cholesky;    
    UpdatableCholesky K_XX_plus_noise_cholesky;
    bool use_cholesky;
    
    // Predefined sets of candidates for efficiency
    std::vector<std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>> predefined_S;
    std::vector<Eigen::MatrixXd> K_XS;
    std::vector<Eigen::MatrixXd> precomputed_K_tilde_XX_inv_times_K_XS;

};

template <int d>
Gaussian_Process<d>::Gaussian_Process(const MeanFunction<d>& mean_function, const KernelFunction<d>& covariance_function)
{
    this->m_unconditioned_mean_function = mean_function.new_clone();
    this->m_unconditioned_covariance_function =  covariance_function.new_clone();
    this->use_cholesky = true;
}

template <int d>
void Gaussian_Process<d>::reset(const MeanFunction<d>& mean_function, const KernelFunction<d>& covariance_function, const std::vector<Gaussian_Process_Observation<d>>& known_values)
{
    // We update the mean and kernel functions
    delete this->m_unconditioned_mean_function;
    this->m_unconditioned_mean_function =  mean_function.new_clone();
    delete this->m_unconditioned_covariance_function;
    this->m_unconditioned_covariance_function = covariance_function.new_clone();
    
    // Reset known values
    this->setKnownValues(known_values);
    
}

template <int d>
bool Gaussian_Process<d>::is_kernel_sparse() const
{
    return (this->m_unconditioned_covariance_function->is_sparse());
}

template <int d>
Gaussian_Process<d>::~Gaussian_Process()
{
    delete this->m_unconditioned_mean_function;
    delete this->m_unconditioned_covariance_function;
}

template <int d>
void Gaussian_Process<d>::reestimate_unconditioned_mean(const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& known_values)
{
    delete this->m_unconditioned_mean_function;
    Translation<d> translation(Eigen::Matrix<double,d,1>::Zero());
    translation.fit(known_values);
    this->m_unconditioned_mean_function = translation.new_clone();
}

template <int d>
void Gaussian_Process<d>::mean(Eigen::Matrix<double,d,1>& res, const Eigen::Matrix<double,d,1>& x)
{    
    res = this->m_unconditioned_mean_function->operator()(x);
    int nb_known_values = (int)m_known_values.size();
    if (nb_known_values>0)
    {
        // Compute the matrix K_X_tr
        for (int k=0; k<nb_known_values; ++k)
            preallocated_matrix_for_K_X_tr.block(0,d*k,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[k].input_point,x);
        
        res = res + preallocated_matrix_for_K_X_tr*precomputed_matrix_product_for_mean;
        
 
    }
    
    
   // std::cout << x << " " << res.transpose() << std::endl;
}

template <int d>
void Gaussian_Process<d>::covariance(Eigen::Matrix<double,d,d>& res, const Eigen::Matrix<double,d,1>& x, const Eigen::Matrix<double,d,1>& x_prime)
{
    // Unconditioned part
    res = this->m_unconditioned_covariance_function->operator()(x,x_prime);
    
    int nb_known_values = (int)m_known_values.size();
    if (nb_known_values>0)
    {
        // Compute the matrices K_X_tr and K_X_prime
        for (int k=0; k<nb_known_values; ++k)
        {
            preallocated_matrix_for_K_X_tr.block(0,d*k,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[k].input_point,x);
            if (x!=x_prime)
                preallocated_matrix_for_K_X_prime.block(d*k,0,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[k].input_point,x_prime);
        }
        if (x==x_prime)
             preallocated_matrix_for_K_X_prime = preallocated_matrix_for_K_X_tr.transpose();
        
        // Using inverse
     //   Eigen::MatrixXd res2 = res - preallocated_matrix_for_K_X_tr*K_XX_plus_noise_inv*preallocated_matrix_for_K_X_prime;
     //   std::cout << "Using inverse: " << res2 << std::endl;
        
        // Using Cholesky decomposition
        if (this->use_cholesky)
            res = res - preallocated_matrix_for_K_X_tr*(K_XX_plus_noise_cholesky.solve(preallocated_matrix_for_K_X_prime));
        else
            res = res - preallocated_matrix_for_K_X_tr*K_XX_plus_noise_inv*preallocated_matrix_for_K_X_prime;
 //       std::cout << "Using Cholesky: " << res << std::endl;
    }
}

template <int d>
void Gaussian_Process<d>::predict_joint_covariance(Eigen::MatrixXd& res, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations)
{    
    // Unconditioned part
    this->predict_joint_unconditioned_covariance(res,target_locations);
    
    // Add the conditioned part
    this->predict_joint_covariance(res,res,target_locations);
}

template <int d>
void Gaussian_Process<d>::predict_joint_covariance(Eigen::MatrixXd& res, const Eigen::MatrixXd& unconditioned_covariance, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations)
{
    int nb_known_values = (int)m_known_values.size();
    int nb_target_locations = target_locations.size();
    if (nb_known_values>0)
    {
        // Resize the matrices
        preallocated_matrix_for_K_X_tr.resize(d*nb_target_locations,d*nb_known_values);
        
        // Compute the matrices K_X_tr and K_X_prime
        typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_locations;
        for (int k=0; k<nb_known_values; ++k)
        {
            int i(0);
            for (it_locations=target_locations.begin(); it_locations!=target_locations.end(); ++it_locations)
            {
                preallocated_matrix_for_K_X_tr.block(d*i,d*k,d,d) = this->m_unconditioned_covariance_function->operator()(*it_locations,m_known_values[k].input_point);
                ++i;
            }
        }
        preallocated_matrix_for_K_X_prime = preallocated_matrix_for_K_X_tr.transpose();
        
        if (this->use_cholesky)
            res = unconditioned_covariance - preallocated_matrix_for_K_X_tr*(K_XX_plus_noise_cholesky.solve(preallocated_matrix_for_K_X_prime));
        else
            res = unconditioned_covariance - preallocated_matrix_for_K_X_tr*K_XX_plus_noise_inv*preallocated_matrix_for_K_X_prime;
        
        preallocated_matrix_for_K_X_tr.resize(d,d*nb_known_values);
        preallocated_matrix_for_K_X_prime.resize(d*nb_known_values,d);
    }
    else
        res = unconditioned_covariance;
  
}

template <int d>
void Gaussian_Process<d>::predict_joint_unconditioned_covariance(Eigen::MatrixXd& res, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations)
{
    int nb_target_locations = target_locations.size();
    Eigen::MatrixXd temp_block(d,d);
    res.resize(d*nb_target_locations,d*nb_target_locations);
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_locations_i, it_locations_j;
    int i(0);
    for (it_locations_i=target_locations.begin(); it_locations_i!=target_locations.end(); ++it_locations_i)
    {
        //std::cout << i+1 << "/" << nb_locations << " ";
        int j(i);
        for (it_locations_j=it_locations_i; it_locations_j!=target_locations.end(); ++it_locations_j)
        {
            temp_block = this->m_unconditioned_covariance_function->operator()(*it_locations_i,*it_locations_j);
            res.block(d*i,d*j,d,d) = temp_block;
            if (i<j)
                res.block(d*j,d*i,d,d) = temp_block.transpose();
            ++j;
        }
        ++i;
    }
}


// Not necessarily square
template <int d>
void Gaussian_Process<d>::create_arbitrary_covariance_matrix(Eigen::MatrixXd& K, const std::vector<Eigen::Matrix<double,d,1>>& input_locations_i, const std::vector<Eigen::Matrix<double,d,1>>& input_locations_j, bool use_unconditioned_covariance_function)
{
    int nb_locations_i = input_locations_i.size();
    int nb_locations_j = input_locations_j.size();
    Eigen::MatrixXd temp_block(d,d);
    K.resize(d*nb_locations_i,d*nb_locations_j);
    for (int i=0; i<nb_locations_i; ++i)
    {
        for (int j=0; j<nb_locations_j; ++j)
        {
            if (use_unconditioned_covariance_function)
                this->unconditioned_covariance(temp_block,input_locations_i[i],input_locations_j[j]);
            else
                this->covariance(temp_block,input_locations_i[i],input_locations_j[j]);
            K.block(d*i,d*j,d,d) = temp_block;
        }
    }
}


// This recomputes everything from scratch
template <int d>
void Gaussian_Process<d>::setKnownValues(const std::vector<Gaussian_Process_Observation<d>>& known_values)
{
    int nb_known_values = known_values.size();
    if (nb_known_values==0)
        this->clearKnownValues();
    else
    {
    
    m_known_values = known_values;
    
    Eigen::MatrixXd temp_block;
    
    // Resize the preallocated matrices
    preallocated_matrix_for_K_X_tr.resize(d,nb_known_values*d);
    preallocated_matrix_for_K_X_prime.resize(nb_known_values*d,d);
    K_XX.resize(nb_known_values*d,nb_known_values*d);
    observation_noise_matrix.resize(nb_known_values*d,nb_known_values*d);
    Y_minus_mu_X.resize(nb_known_values*d);
    
    // Fill the matrices
    observation_noise_matrix.setIdentity();
    for (int k=0; k<nb_known_values; ++k)
    {
        // Observation noise
        observation_noise_matrix.block(d*k,d*k,d,d) = m_known_values[k].observation_noise_covariance_matrix;
     //   observation_noise_matrix(d*k+1,d*k+1) = m_known_values[k].observation_noise_variance;
        
        // Y minus mu(X)
        Eigen::Matrix<double,d,1> current_point = m_known_values[k].output_point - this->m_unconditioned_mean_function->operator()(m_known_values[k].input_point);
        for (int i=0; i<d; ++i)
            Y_minus_mu_X(d*k+i) =  current_point(i);

        // K_XX
        for (int l=k; l<nb_known_values; ++l)
        {
            K_XX.block(d*k,d*l,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[k].input_point,m_known_values[l].input_point);
            if (l>k)
            {
                temp_block = K_XX.block(d*k,d*l,d,d);
                K_XX.block(d*l,d*k,d,d) = temp_block.transpose();
            }
        }
    }
    
    
    // Final precomputations

    
    // Cholesky decomposition
    if (this->use_cholesky)
    {
        K_XX_plus_noise_cholesky.compute(K_XX + observation_noise_matrix); 
        
//         std::cout << "Matrix L:" << std::endl;
//         Eigen::MatrixXd temp;
//         temp = K_XX_plus_noise_cholesky.matrixL();
//         std::cout << temp << std::endl;
//         std::cout << "Matrix U:" << std::endl;
//         temp = K_XX_plus_noise_cholesky.matrixU();
//         std::cout << temp << std::endl;
//         std::cout << "Matrix LLT:" << std::endl;
//         temp = K_XX_plus_noise_cholesky.matrixLLT();
//         std::cout << temp << std::endl;
//         std::cout << "KXX:" << std::endl;
//         std::cout << K_XX + observation_noise_matrix << std::endl;
        precomputed_matrix_product_for_mean = K_XX_plus_noise_cholesky.solve(Y_minus_mu_X);
       // cv::waitKey(1000);
    }
    else
    {
        K_XX_plus_noise_inv = (K_XX + observation_noise_matrix).inverse();
        precomputed_matrix_product_for_mean = K_XX_plus_noise_inv*Y_minus_mu_X;
    }
    
    int nb_predefined_sets = predefined_S.size();
    for (int ind_S = 0; ind_S<nb_predefined_sets; ++ind_S)
    {
        this->compute_unconditioned_K_XT(K_XS[ind_S],predefined_S[ind_S]);
        if (this->use_cholesky)
            precomputed_K_tilde_XX_inv_times_K_XS[ind_S] = K_XX_plus_noise_cholesky.solve(K_XS[ind_S]);
        else
            precomputed_K_tilde_XX_inv_times_K_XS[ind_S] = K_XX_plus_noise_inv*K_XS[ind_S];
    }
    
    
    }
    // A test
  //  std::cout << "Known value : " << m_known_values[0].input_point << " " << m_known_values[0].output_point << std::endl;
  //  Eigen::VectorXd test;
  //  this->mean(test,m_known_values[0].input_point);
  //  std::cout << "Predicted value : " << test.transpose() << std::endl;
}

// Efficient update
template <int d>
void Gaussian_Process<d>::addKnownValue(const Gaussian_Process_Observation<d>& obs)
{
    m_known_values.push_back(obs);
    int nb_known_values = m_known_values.size();
    
    if (nb_known_values==1)
        this->setKnownValues(std::vector<Gaussian_Process_Observation<d>>(1,obs));
    else
    {
        
    
     // Resize the preallocated matrices
    preallocated_matrix_for_K_X_tr.resize(d,nb_known_values*d);
    preallocated_matrix_for_K_X_prime.resize(nb_known_values*d,d);
    K_XX.conservativeResize(nb_known_values*d,nb_known_values*d);
    observation_noise_matrix.conservativeResizeLike(Eigen::MatrixXd::Zero(nb_known_values*d,nb_known_values*d));
    Y_minus_mu_X.conservativeResize(nb_known_values*d);
    
    int ind = nb_known_values-1;
    
    // Update Y minus mu(X)
    Eigen::Matrix<double,d,1> new_point = obs.output_point - this->m_unconditioned_mean_function->operator()(obs.input_point);
    for (int i=0; i<d; ++i)
        Y_minus_mu_X(d*ind + i) =  new_point(i);
    

    // Update K_XX and the observation noise_matrix
    observation_noise_matrix.block(d*ind,d*ind,d,d) = m_known_values[ind].observation_noise_covariance_matrix;
    Eigen::MatrixXd temp_block;
    for (int l=0; l<nb_known_values; ++l)
    {
        K_XX.block(d*ind,d*l,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[ind].input_point,m_known_values[l].input_point);
        if (l!=ind)
        {
            temp_block = K_XX.block(d*ind,d*l,d,d);
            K_XX.block(d*l,d*ind,d,d) = temp_block.transpose();
        }
    }
    
    

        Eigen::MatrixXd K_XX_tilde = K_XX + observation_noise_matrix; // new matrix
        Eigen::MatrixXd R = K_XX_tilde.block(ind*d,0,d,ind*d);
        Eigen::MatrixXd Q = R.transpose();
        Eigen::MatrixXd S = K_XX_tilde.block(ind*d,ind*d,d,d);
        UpdatableCholesky P_cholesky = K_XX_plus_noise_cholesky;
        if (this->use_cholesky)
        {
            
           // K_XX_plus_noise_cholesky.compute(K_XX + observation_noise_matrix); 
            K_XX_plus_noise_cholesky.update(Q,S,K_XX_tilde);
            precomputed_matrix_product_for_mean = K_XX_plus_noise_cholesky.solve(Y_minus_mu_X);
        }
        else
        {
     //       std::cout << "Updates" << std::endl;
 
            // Update inverse
            Eigen::MatrixXd current_inv = K_XX_plus_noise_inv; 
            efficient_update_matrix_inverse(K_XX_plus_noise_inv,current_inv,Q,R,S);
           
            // Update precomputed matrix product for mean
            Eigen::MatrixXd current_precomputed_matrix_product_for_mean = precomputed_matrix_product_for_mean; // not symmetric in general
            Eigen::MatrixXd new_row_Y_minus_mu_X = Y_minus_mu_X.block(d*ind,0,d,1);
            efficient_update_matrix_inverse_times_B(precomputed_matrix_product_for_mean,current_precomputed_matrix_product_for_mean,new_row_Y_minus_mu_X,R,K_XX_plus_noise_inv);
        }
        
        // Update inverse * K_XS
        int nb_predefined_sets = predefined_S.size();
    
        for (int ind_S = 0; ind_S<nb_predefined_sets; ++ind_S)
        {
                
            // Update K_XS
            int size_S = predefined_S[ind_S].size();
            Eigen::MatrixXd new_row_K_XS(d,d*size_S);
            typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
            int j(0);
            for (it_candidate = predefined_S[ind_S].begin(); it_candidate!=predefined_S[ind_S].end(); ++it_candidate)
            {
                new_row_K_XS.block(0,d*j,d,d) = this->m_unconditioned_covariance_function->operator()(obs.input_point,*it_candidate);
                ++j;
            }
            K_XS[ind_S].conservativeResize(d*nb_known_values,d*size_S);
            K_XS[ind_S].block(d*ind,0,d,d*size_S) = new_row_K_XS;
            
            Eigen::MatrixXd current_K_tilde_XX_inv_times_K_XS = precomputed_K_tilde_XX_inv_times_K_XS[ind_S];
                
            if (this->use_cholesky)
                efficient_update_matrix_inverse_times_B(precomputed_K_tilde_XX_inv_times_K_XS[ind_S],current_K_tilde_XX_inv_times_K_XS,new_row_K_XS,Q,S,P_cholesky);
            else
                efficient_update_matrix_inverse_times_B(precomputed_K_tilde_XX_inv_times_K_XS[ind_S],current_K_tilde_XX_inv_times_K_XS,new_row_K_XS,R,K_XX_plus_noise_inv);
        }
            
//             std::cout << "Sanity check" << std::endl;
//             Eigen::MatrixXd temp;
//          //   temp = K_XX_plus_noise_inv - K_XX_tilde.inverse();
//      //       std::cout << temp.maxCoeff() << std::endl;
//             for (int ind_S = 0; ind_S<nb_predefined_sets; ++ind_S)
//             {
//                 temp = precomputed_K_tilde_XX_inv_times_K_XS[ind_S] - K_XX_plus_noise_cholesky.solve(K_XS[ind_S]);
//                 std::cout << temp.maxCoeff() << std::endl;
//             }
//              std::cout << "Done" << std::endl;
    }    
}


template <int d>
void Gaussian_Process<d>::clearKnownValues()
{
    m_known_values.clear();
    K_XX.resize(0,0);
    observation_noise_matrix.resize(0,0);
    Y_minus_mu_X.resize(0);
    precomputed_matrix_product_for_mean.resize(0,0);
    preallocated_matrix_for_K_X_tr.resize(0,0);
    preallocated_matrix_for_K_X_prime.resize(0,0);
    K_XX_plus_noise_cholesky = UpdatableCholesky();
}

template <int d>
void Gaussian_Process<d>::reestimate_covariance_hyperparameters(const std::vector<Gaussian_Process_Observation<d>>& known_values)
{
   //  m_unconditioned_covariance_function->estimateKernelHyperparametersGPP_alternate(known_values,*(this->m_unconditioned_mean_function));
   //  m_unconditioned_covariance_function->estimateKernelHyperparametersGPP(known_values,*(this->m_unconditioned_mean_function));
    m_unconditioned_covariance_function->estimateKernelHyperparametersLOO(known_values,*(this->m_unconditioned_mean_function));
}

template <int d>
void Gaussian_Process<d>::precompute_helper_for_entropy_decrease_computation(Helper_Entropy_Decrease_Computation& helper, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations)
{
    // Vectorise values of interest
    int nb_known_values = m_known_values.size();
    int nb_target_locations = target_locations.size();
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_target;
    std::vector<Eigen::Matrix<double,d,1>> X_vec(nb_known_values), T_vec(nb_target_locations);
    for (int ind_x=0; ind_x<nb_known_values; ++ind_x)
        X_vec[ind_x] = m_known_values[ind_x].input_point;
    int ind_T(0);
    for (it_target = target_locations.begin(); it_target!=target_locations.end(); ++it_target)
    {
        T_vec[ind_T] = *it_target;
        ++ind_T;
    }
    
    // Compute helper
    Eigen::MatrixXd K_T_T;
    this->create_arbitrary_covariance_matrix(K_T_T,T_vec,T_vec,true);
    helper.K_T_T_cholesky = K_T_T.llt();
    
    if (nb_known_values>0)
    {
        Eigen::MatrixXd A, K_X_T, K_X_T_tr;
        this->create_arbitrary_covariance_matrix(K_X_T,X_vec,T_vec,true);
        K_X_T_tr = K_X_T.transpose();
        helper.K_T_T_inv_times_K_X_T_tr = (helper.K_T_T_cholesky).solve(K_X_T_tr);
        A = K_XX + observation_noise_matrix - K_X_T*(helper.K_T_T_inv_times_K_X_T_tr);
        helper.A_cholesky = A.llt();
    }
}

template <int d>
double Gaussian_Process<d>::compute_entropy_decrease(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations, const Eigen::Matrix<double,d,1>& added_location, const Helper_Entropy_Decrease_Computation& helper)
{
    Eigen::Matrix<double,d,d> I_d = Eigen::Matrix<double,d,d>::Identity();
    
    // Vectorise values of interest
    int nb_known_values = m_known_values.size();
    int nb_target_locations = target_locations.size();
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_target;
    std::vector<Eigen::Matrix<double,d,1>> X_vec(nb_known_values), T_vec(nb_target_locations);
    for (int ind_x=0; ind_x<nb_known_values; ++ind_x)
        X_vec[ind_x] = m_known_values[ind_x].input_point;
    int ind_T(0);
    for (it_target = target_locations.begin(); it_target!=target_locations.end(); ++it_target)
    {
        T_vec[ind_T] = *it_target;
        ++ind_T;
    }
    
    
    // Not precomputable
    Eigen::MatrixXd K_Xnplus1_T, K_Xnplus1_X, K_Xnplus1_T_tr, var_added_location;
    this->create_arbitrary_covariance_matrix(K_Xnplus1_T,std::vector<Eigen::Matrix<double,d,1>>(1,added_location),T_vec,true);
    K_Xnplus1_T_tr = K_Xnplus1_T.transpose();
    this->create_arbitrary_covariance_matrix(var_added_location,std::vector<Eigen::Matrix<double,d,1>>(1,added_location),std::vector<Eigen::Matrix<double,d,1>>(1,added_location),true);
    
    Eigen::MatrixXd lower_matrix, upper_matrix;
    if (nb_known_values==0)
    {    
   //     upper_matrix = var_added_location + observation_noise*I_d;
   //     lower_matrix = upper_matrix - K_Xnplus1_T*((helper.K_T_T_cholesky).solve(K_Xnplus1_T_tr)); 
        
        // New one, where the observation noise is 0
        upper_matrix = var_added_location;
        lower_matrix = upper_matrix - K_Xnplus1_T*((helper.K_T_T_cholesky).solve(K_Xnplus1_T_tr));
    }
    else
    {

        // Non precomputable
        Eigen::MatrixXd K_Xnplus1_X, C, C_tr, var_added_location_conditioned;
        this->create_arbitrary_covariance_matrix(var_added_location_conditioned,std::vector<Eigen::Matrix<double,d,1>>(1,added_location),std::vector<Eigen::Matrix<double,d,1>>(1,added_location),false);
        this->create_arbitrary_covariance_matrix(K_Xnplus1_X,std::vector<Eigen::Matrix<double,d,1>>(1,added_location),X_vec,true);
        C = K_Xnplus1_X - K_Xnplus1_T*(helper.K_T_T_inv_times_K_X_T_tr);
        C_tr = C.transpose();        
    
        // Previous one
    //    upper_matrix = var_added_location_conditioned + observation_noise*I_d;
   //     lower_matrix = var_added_location + observation_noise*I_d - K_Xnplus1_T*((helper.K_T_T_cholesky).solve(K_Xnplus1_T_tr)) - C*((helper.A_cholesky).solve(C_tr));
        
        // New one, where the observation noise is 0
        upper_matrix = var_added_location_conditioned;
        lower_matrix = var_added_location - K_Xnplus1_T*((helper.K_T_T_cholesky).solve(K_Xnplus1_T_tr)) - C*((helper.A_cholesky).solve(C_tr));
        
    //    std::cout << "Lower matrix: " << std::endl;
    //    std::cout << upper_matrix << std::endl;
    }
    
    double upper_det = upper_matrix.determinant();
    double lower_det = lower_matrix.determinant();
 //   std::cout << "Upper determinant fast: " <<  upper_det << std::endl;
 //   std::cout << "Lower determinant fast: " <<  lower_det << std::endl;
        
    double entropy_decrease = 0.5*(std::log(upper_det) - std::log(lower_det));
    //   std::cout << "Result fast: " <<  entropy_decrease << std::endl;

    return entropy_decrease;
    

}

template <int d>
double Gaussian_Process<d>::compute_entropy_decrease(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations, const Eigen::Matrix<double,d,1>& added_location)
{
    
    Helper_Entropy_Decrease_Computation helper;
    this->precompute_helper_for_entropy_decrease_computation(helper,target_locations);
    return this->compute_entropy_decrease(target_locations,added_location,helper);
}

template <int d>
double Gaussian_Process<d>::compute_entropy(const Eigen::Matrix<double,d,1>& x)
{
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> singleton_x;
    singleton_x.insert(x);
    return (this->compute_entropy(singleton_x));
}

template <int d>
double Gaussian_Process<d>::compute_entropy(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations)
{
    Eigen::MatrixXd cov_matrix;
    this->predict_joint_covariance(cov_matrix,target_locations);
    int size_T = target_locations.size();
    double H = 0.5*d*size_T*(1 + std::log(2*M_PI)) + 0.5*std::log(cov_matrix.determinant());
    return H;
}

template <int d>
void Gaussian_Process<d>::set_predefined_candidate_locations(const std::vector<std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>>& candidate_locations)
{
    int nb_predefined_sets = candidate_locations.size();
    predefined_S.resize(nb_predefined_sets);
    K_XS.resize(nb_predefined_sets);
    precomputed_K_tilde_XX_inv_times_K_XS.resize(nb_predefined_sets);
    for (int ind_S = 0; ind_S<nb_predefined_sets; ++ind_S)
    {
        predefined_S[ind_S] = candidate_locations[ind_S];
        this->compute_unconditioned_K_XT(K_XS[ind_S],predefined_S[ind_S]);
        precomputed_K_tilde_XX_inv_times_K_XS[ind_S] = K_XX_plus_noise_inv*K_XS[ind_S];
    }
}

template <int d>
void Gaussian_Process<d>::set_predefined_candidate_locations(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& candidate_locations)
{
    std::vector<std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>> candidate_locations_vec(1);
    candidate_locations_vec[0] = candidate_locations;
    this->set_predefined_candidate_locations(candidate_locations_vec);
}

template <int d>
void Gaussian_Process<d>::compute_unconditioned_K_XT(Eigen::MatrixXd& K_XT, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& T)
{
    int nb_targets = (int)T.size();
    int nb_known_values = (int)m_known_values.size();
    
    // This could be precomputed beforehand or updated efficiently
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_T;
    int j(0);
    K_XT.resize(d*nb_known_values,d*nb_targets);
    for (it_T = T.begin(); it_T!=T.end(); ++it_T)
    {
       for (int l=0; l<nb_known_values; ++l)
       {
           K_XT.block(d*l,d*j,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[l].input_point,*it_T);
       }
       ++j;
   }
}

template <int d>
void Gaussian_Process<d>::compute_entropies_of_predefined_candidates(std::map<Eigen::Matrix<double,d,1>,double,comp_Point<d>>& entropies, int ind_predefined_candidate_set)
{
    int nb_known_values = (int)m_known_values.size();
    
    // This could be precomputed beforehand or updated efficiently
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
    int j(0);
 //   Eigen::MatrixXd K_XS(d*nb_known_values,d*nb_candidates);
 //   for (it_candidate = candidate_locations.begin(); it_candidate!=candidate_locations.end(); ++it_candidate)
 //   {
 //       for (int l=0; l<nb_known_values; ++l)
 //       {
 //           K_XS.block(d*l,d*j,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[l].input_point,*it_candidate);
 //       }
 //       ++j;
 //   }
    
  //  std::cout << "Product begin" << std::endl;
    //Eigen::MatrixXd A = K_XX_plus_noise_inv*K_XS;
    Eigen::MatrixXd A = precomputed_K_tilde_XX_inv_times_K_XS[ind_predefined_candidate_set];
  //  std::cout << "Product end" << std::endl;
    
    Eigen::MatrixXd cov_matrix;
    j=0;
    entropies.clear();
    for (it_candidate = predefined_S[ind_predefined_candidate_set].begin(); it_candidate!=predefined_S[ind_predefined_candidate_set].end(); ++it_candidate)
    {
        cov_matrix = this->m_unconditioned_covariance_function->operator()(*it_candidate,*it_candidate) - ((K_XS[ind_predefined_candidate_set].block(0,d*j,d*nb_known_values,d)).transpose())*(A.block(0,d*j,d*nb_known_values,d));
        
        double H = 0.5*d*(1 + std::log(2*M_PI)) + 0.5*std::log(cov_matrix.determinant());
        
        entropies.insert(std::pair<Eigen::Matrix<double,d,1>,double>(*it_candidate,H));
        
        ++j;
    }
}

template <int d>
double Gaussian_Process<d>::compute_trace_covariance_predefined_candidates(int ind_predefined_candidate_set)
{
    int nb_known_values = (int)m_known_values.size();
    
    // This could be precomputed beforehand or updated efficiently
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
    int j(0);
 //   Eigen::MatrixXd K_XS(d*nb_known_values,d*nb_candidates);
 //   for (it_candidate = candidate_locations.begin(); it_candidate!=candidate_locations.end(); ++it_candidate)
 //   {
 //       for (int l=0; l<nb_known_values; ++l)
 //       {
 //           K_XS.block(d*l,d*j,d,d) = this->m_unconditioned_covariance_function->operator()(m_known_values[l].input_point,*it_candidate);
 //       }
 //       ++j;
 //   }
    
  //  std::cout << "Product begin" << std::endl;
    //Eigen::MatrixXd A = K_XX_plus_noise_inv*K_XS;
    Eigen::MatrixXd A = precomputed_K_tilde_XX_inv_times_K_XS[ind_predefined_candidate_set];
  //  std::cout << "Product end" << std::endl;
    
    Eigen::MatrixXd cov_matrix;
    j=0;
    double res_trace(0);
    for (it_candidate = predefined_S[ind_predefined_candidate_set].begin(); it_candidate!=predefined_S[ind_predefined_candidate_set].end(); ++it_candidate)
    {
        cov_matrix = this->m_unconditioned_covariance_function->operator()(*it_candidate,*it_candidate) - ((K_XS[ind_predefined_candidate_set].block(0,d*j,d*nb_known_values,d)).transpose())*(A.block(0,d*j,d*nb_known_values,d));
        res_trace += cov_matrix.trace();
        ++j;
    }
    
    return res_trace;
}

template <int d>
void Gaussian_Process<d>::compute_covariance_matrix_at_predefined_candidates(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>,comp_Point<d>>& covariance_matrices, int ind_predefined_candidate_set)
{
    int nb_known_values = (int)m_known_values.size();
    covariance_matrices.clear();
    
    // This could be precomputed beforehand or updated efficiently
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
    int j(0);

    Eigen::MatrixXd A = precomputed_K_tilde_XX_inv_times_K_XS[ind_predefined_candidate_set];
    
    Eigen::Matrix<double,d,d> cov_matrix;
    j=0;
    double res_trace(0);
    for (it_candidate = predefined_S[ind_predefined_candidate_set].begin(); it_candidate!=predefined_S[ind_predefined_candidate_set].end(); ++it_candidate)
    {
        cov_matrix = this->m_unconditioned_covariance_function->operator()(*it_candidate,*it_candidate) - ((K_XS[ind_predefined_candidate_set].block(0,d*j,d*nb_known_values,d)).transpose())*(A.block(0,d*j,d*nb_known_values,d));
        covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it_candidate,cov_matrix));
        ++j;
    }
}

template <int d>
void Gaussian_Process<d>::predict_transformation_at_predefined_candidates(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& predicted_transformation, int ind_predefined_candidate_set)
{
    int nb_known_values = (int)m_known_values.size();
    
    // This could be precomputed beforehand or updated efficiently
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
    int j(0);
 
    // We use the precomputed part (we could probably even update A efficiently)
    Eigen::MatrixXd A = (K_XS[ind_predefined_candidate_set].transpose())*precomputed_matrix_product_for_mean; // this should be a vector
    if (A.cols()!=1)
        std::cout << "A should be a vector!" << std::endl;
    
    j=0;
    predicted_transformation.clear();
    for (it_candidate = predefined_S[ind_predefined_candidate_set].begin(); it_candidate!=predefined_S[ind_predefined_candidate_set].end(); ++it_candidate)
    {
        Eigen::Matrix<double,d,1> unconditioned_point = this->m_unconditioned_mean_function->operator()(*it_candidate);
        Eigen::Matrix<double,d,1> conditioned_part = A.block(j*d,0,d,1);
        Eigen::Matrix<double,d,1> predicted_point = unconditioned_point + conditioned_part;
        predicted_transformation.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(*it_candidate,predicted_point));
        
        ++j;
    }
  
}

template <int d>
void Gaussian_Process<d>::evaluate_quality(std::ofstream& infile, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_locations)
{
    
    int nb_locations = target_locations.size();    
 //   cv::Mat prediction_errors(nb_locations,1,CV_64F);
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it;
  //convert_cv_point_to_eigen(  Eigen::Matrix<double,d,1> predicted_mean, error;
    Eigen::Matrix<double,d,1> predicted_mean, true_value, error;
    Eigen::Matrix<double,d,d> predicted_covariance_x;
    
    
    // Evaluation measures
    int nb_eval_measures = 5;
    std::vector<double> eval_measures(nb_eval_measures,0); // MAD, RMSD, Max error, NLPP, GPE (both one variable)
    double error_norm;
    
    for (it=target_locations.begin(); it!=target_locations.end(); ++it)
    {
        // Target location
        Eigen::Matrix<double,d,1> x = *it;
        
        //std::cout << "Mean" << std::endl;
        // Predicted mean at x
        this->mean(predicted_mean,x);
        
        //std::cout << "Covariance" << std::endl;
        // Predicted covariance at x
        this->covariance(predicted_covariance_x,x,x);
       
        // Displacement
        true_value = true_transformation.at(*it);
        
        error = true_value - predicted_mean;
        
        // L2 norm
        error_norm = error.norm();
        
        // L1 norm
        eval_measures[0] += error_norm ;
        
        // L2 norm
        eval_measures[1] +=  error_norm*error_norm;
        
        // L-inf
        if (error_norm>eval_measures[2])
            eval_measures[2] = error_norm; 
            
        // Negative log-predictive probability (= negative log likelihood of the truth, as observation, given the GP)
        eval_measures[3] += 0.5*(error.transpose())*(predicted_covariance_x.inverse())*error;
        eval_measures[3] += std::log(2*M_PI);
        eval_measures[3] += 0.5*std::log(predicted_covariance_x.determinant());
        
        // GPE
        eval_measures[4] += error_norm*error_norm + predicted_covariance_x.trace();
    }
    
    infile << nb_locations << " ";
    for (int k=0; k<nb_eval_measures; ++k)
        infile << eval_measures[k] << " ";
    infile << std::endl;

}

template <int d>
void Gaussian_Process<d>::evaluate_quality_on_predefined_set(std::ofstream& infile, std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& predicted_transformation, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, int ind_predefined_candidate_set)
{
    int nb_known_values = (int)m_known_values.size();
    int nb_locations = predefined_S[ind_predefined_candidate_set].size();  
    predicted_transformation.clear();
    
    // Evaluation measures
    int nb_eval_measures = 5;
    std::vector<double> eval_measures(nb_eval_measures,0); // MAD, RMSD, Max error, NLPP, GPE (both one variable)
    double error_norm;
    
      
    // Precomputed matrices
    Eigen::MatrixXd A_mean = (K_XS[ind_predefined_candidate_set].transpose())*precomputed_matrix_product_for_mean;
    Eigen::MatrixXd A_cov = precomputed_K_tilde_XX_inv_times_K_XS[ind_predefined_candidate_set];
     
    // Iteration over points
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
    int j(0);
    Eigen::Matrix<double,d,1> predicted_mean, true_value, error;
    Eigen::Matrix<double,d,d> predicted_covariance_x;
    for (it_candidate = predefined_S[ind_predefined_candidate_set].begin(); it_candidate!=predefined_S[ind_predefined_candidate_set].end(); ++it_candidate)
    {
        // Predicted covariance matrix
        predicted_covariance_x = this->m_unconditioned_covariance_function->operator()(*it_candidate,*it_candidate) - ((K_XS[ind_predefined_candidate_set].block(0,d*j,d*nb_known_values,d)).transpose())*(A_cov.block(0,d*j,d*nb_known_values,d));
        
        // Predicted mean
        predicted_mean = this->m_unconditioned_mean_function->operator()(*it_candidate) + A_mean.block(j*d,0,d,1);
        
       // std::cout << "Predicted value: " << predicted_mean.transpose();
        
        predicted_transformation.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(*it_candidate,predicted_mean));
        
        // Displacement
        true_value = true_transformation.at(*it_candidate);
        
       // std::cout << " -- True value: " << true_value.transpose() << std::endl;
        
        error = true_value - predicted_mean;
        
        // L2 norm
        error_norm = error.norm();
        
        // L1 norm
        eval_measures[0] += error_norm ;
        
        // L2 norm
        eval_measures[1] +=  error_norm*error_norm;
        
        // L-inf
        if (error_norm>eval_measures[2])
            eval_measures[2] = error_norm; 
            
        // Negative log-predictive probability (= negative log likelihood of the truth, as observation, given the GP)
        eval_measures[3] += 0.5*(error.transpose())*(predicted_covariance_x.inverse())*error;
        eval_measures[3] += std::log(2*M_PI);
        eval_measures[3] += 0.5*std::log(predicted_covariance_x.determinant());
        
        // GPE
        eval_measures[4] += error_norm*error_norm + predicted_covariance_x.trace();
        if (predicted_covariance_x.trace()<0)
            std::cout << predicted_covariance_x << std::endl;
        
        ++j;
    }
    
    // Write on file
    infile << nb_locations << " ";
    for (int k=0; k<nb_eval_measures; ++k)
        infile << eval_measures[k] << " ";
    infile << std::endl;

}

template <int d>
void Gaussian_Process<d>::evaluate_transformations_on_predefined_set(std::vector<std::vector<double>>& transformation_scores, const std::vector<std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>>& transformations_to_evaluate, int ind_predefined_candidate_set, EvaluationMeasure eval_measure)
{
    int nb_known_values = (int)m_known_values.size();
    int nb_locations = predefined_S[ind_predefined_candidate_set].size();  
    
    // Evaluation measures
    int nb_eval_measures = 2;
    transformation_scores.resize(nb_eval_measures); // MAD, RMSD, Max error, NLPP, GPE (both one variable)
    int nb_transformations = (int)transformations_to_evaluate.size();
    for (int m=0; m<nb_eval_measures; ++m)
        transformation_scores[m] = std::vector<double>(nb_transformations,0);
        
    
    double difference_norm;
    
    // Precomputed matrices
    Eigen::MatrixXd A_mean = (K_XS[ind_predefined_candidate_set].transpose())*precomputed_matrix_product_for_mean;
    Eigen::MatrixXd A_cov = precomputed_K_tilde_XX_inv_times_K_XS[ind_predefined_candidate_set];
    
    
    // Iteration over points
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_candidate;
    int j(0);
    Eigen::Matrix<double,d,1> predicted_mean;
    Eigen::Matrix<double,d,d> predicted_covariance_x;
    for (it_candidate = predefined_S[ind_predefined_candidate_set].begin(); it_candidate!=predefined_S[ind_predefined_candidate_set].end(); ++it_candidate)
    {
        // Predicted covariance matrix
        if (nb_known_values>0)
        {
            predicted_covariance_x = this->m_unconditioned_covariance_function->operator()(*it_candidate,*it_candidate) - ((K_XS[ind_predefined_candidate_set].block(0,d*j,d*nb_known_values,d)).transpose())*(A_cov.block(0,d*j,d*nb_known_values,d));
            
            // Predicted mean
            predicted_mean = this->m_unconditioned_mean_function->operator()(*it_candidate) + A_mean.block(j*d,0,d,1);
        }
        else
        {
            predicted_covariance_x = this->m_unconditioned_covariance_function->operator()(*it_candidate,*it_candidate);
            predicted_mean = this->m_unconditioned_mean_function->operator()(*it_candidate);
        }
        
        
        for (int t=0; t<nb_transformations; ++t)
        {
            
            Eigen::Matrix<double,d,1> transformation_value, difference_value;
            
            // Displacement
            transformation_value = transformations_to_evaluate[t].at(*it_candidate);
            difference_value = transformation_value - predicted_mean;
        
            // L2 norm
            difference_norm = difference_value.norm();
            
            // Pick what to do depending on the evaluation measure
            if (eval_measure==l1)
                transformation_scores[0][t] +=  difference_norm;
            if (eval_measure==l2)
                transformation_scores[0][t] +=  difference_norm*difference_norm;
            if (eval_measure==l_inf)
                transformation_scores[0][t] =  std::max<double>(transformation_scores[0][t],difference_norm);
        
            // Negative log-predictive probability (= negative log likelihood of the truth, as observation, given the GP)
            double individual_log_likelihood(0);
            individual_log_likelihood += 0.5*(difference_value.transpose())*(predicted_covariance_x.inverse())*difference_value;
  //          individual_log_likelihood += std::log(2*CV_PI);
  //          individual_log_likelihood += 0.5*std::log(predicted_covariance_x.determinant());
            transformation_scores[1][t] += individual_log_likelihood;

        }
        ++j;
    }
    
    if (eval_measure==l1)
    {
        for (int t=0; t<nb_transformations; ++t)
            transformation_scores[0][t] = transformation_scores[0][t]/((double)nb_locations);
    }
    if (eval_measure==l2)
    {
        for (int t=0; t<nb_transformations; ++t)
        {
            transformation_scores[0][t] = transformation_scores[0][t]/((double)nb_locations);
            transformation_scores[0][t] = std::sqrt(transformation_scores[0][t]);
        }
    }
    
//     std::cout << "Likelihood scores: "; 
//     for (int t=0; t<nb_transformations; ++t)
//         std::cout << transformation_scores[1][t] << " ";
//     std::cout << std::endl;
}

#endif // REGISTRATION_AVERAGING_INCLUDED
