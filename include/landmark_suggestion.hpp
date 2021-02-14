#ifndef LANDMARK_SUGGESTION_INCLUDED
#define LANDMARK_SUGGESTION_INCLUDED

#include <vector>
#include <math.h>
#include <set>
#include <fstream>
#include <Eigen/Dense>
#include "kernel_function.hpp"
#include "kernel_hyperparameter_estimation.hpp"
#include "mean_function.hpp"
#include "wendland_kernel.hpp"
#include "kernel_bundle.hpp"
#include "querying_strategy.hpp"
#include "gp_observation_database.hpp"
#include "updatable_cholesky.hpp"
#include "gaussian_process.hpp"
#include "visualization_functions.hpp"
#include "landmark_annotator.hpp"
#include "partial_window.hpp"



struct RankingMeasures
{
    double spearman_correlation_coefficient;
    double rank_best_true_registration; // where the evaluation measure ranked the actual best transformation: 1 is the best case, the lower the better
    double rank_best_predicted_registration; // what is the true rank of the transformation that was predicted as the best. 1 is the best case, the lower the better
    double pearson_correlation_coefficient;
    double mean_abs_error_on_score;
};


struct Displacement_Measures
{
    int nb_points;
    double l_1; // sum of the absolute displacements
    double l_2; // sum of the squared displacements
    double l_inf; // max of the displacements
};


class Settings
{
public:
    Settings(int input_annotation_budget) :  annotation_budget(input_annotation_budget) {};
    int annotation_budget;
};

class IOSettings
{
    
public:
    IOSettings(const std::string& input_results_folder, const std::string& input_pair_identifier, const std::string& input_experiment_identifier) :
        results_folder(input_results_folder), pair_identifier(input_pair_identifier),  experiment_identifier(input_experiment_identifier) {};

    std::string results_folder;
    std::string pair_identifier;
    std::string experiment_identifier;
};

template <int d>
class InputLocations
{
    
public:
    InputLocations(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& input_candidate_locations,
                        const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& input_target_locations,
                        const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& input_evaluation_locations,
                        const std::vector<Eigen::Matrix<double,d,1>>& input_queries_for_burn_in = std::vector<Eigen::Matrix<double,d,1>>()) :
                        candidate_locations(input_candidate_locations),
                        target_locations(input_target_locations), 
                        evaluation_locations(input_evaluation_locations),
                        queries_for_burn_in(input_queries_for_burn_in) {};
    
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> candidate_locations; // mandatory
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> target_locations; // optional (will be set to candidate_locations by default, which is equvalent to using the whole image domain)
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> evaluation_locations; // optional (empty by default)
    std::vector<Eigen::Matrix<double,d,1>> queries_for_burn_in; // optional (empty by default)
};

template <int d>
class InputTransformations
{
public:
    InputTransformations(const std::vector<std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>>& input_transformations_to_evaluate, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& input_true_transformation) : transformations_to_evaluate(input_transformations_to_evaluate), true_transformation(input_true_transformation) {};
    std::vector<std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>> transformations_to_evaluate;
    std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>> true_transformation;
};



template <int d>
class GaussianProcessPrior
{
    
public:
    GaussianProcessPrior(std::shared_ptr<MeanFunction<d>> input_mean, std::shared_ptr<KernelFunction<d>> input_kernel, const std::vector<std::vector<Gaussian_Process_Observation<d>>>& input_prior_observations);
    std::shared_ptr<MeanFunction<d>> mean;
    std::shared_ptr<KernelFunction<d>> kernel;
    std::vector<std::vector<Gaussian_Process_Observation<d>>> prior_observations;
    
};

template <int d>
GaussianProcessPrior<d>::GaussianProcessPrior(std::shared_ptr<MeanFunction<d>> input_mean, std::shared_ptr<KernelFunction<d>> input_kernel, const std::vector<std::vector<Gaussian_Process_Observation<d>>>& input_prior_observations)
{
    mean = input_mean;
    kernel = input_kernel;
    prior_observations = input_prior_observations;
}


template <typename T, typename comp_T>
bool is_A_in_B(const std::set<T,comp_T>& A, const std::set<T,comp_T>& B)
{
    typename std::set<T,comp_T>::const_iterator it_A;
    it_A = A.begin();
    while ((it_A!=A.end()) && (B.count(*it_A)))
        ++it_A;
    return (it_A==A.end());
}



template <typename T, typename comp_T> 
void union_sets(std::set<T,comp_T>& s_union, const std::set<T,comp_T>& s1, const std::set<T,comp_T>& s2)
{
    s_union.clear();
    typename std::set<T,comp_T>::const_iterator it;
    for (it=s1.begin(); it!=s1.end(); ++it)
        s_union.insert(*it);
    for (it=s2.begin(); it!=s2.end(); ++it)
        s_union.insert(*it);
}


template <typename T, typename comp_T> 
void add_to_set(std::set<T,comp_T>& s_1, const std::set<T,comp_T>& s2)
{
    typename std::set<T,comp_T>::const_iterator it;
    for (it=s2.begin(); it!=s2.end(); ++it)
        s_1.insert(*it);
}

template <typename T, typename comp_T>
void get_input_set(std::set<T,comp_T>& input_set, const std::map<T,T,comp_T>& map)
{
    input_set.clear();
    for (const auto& pt : map)
        input_set.insert(pt.first);
}


template <typename T>
bool compare_pair_wrt_first(const std::pair<T,int>& pair_1, const std::pair<T,int>& pair_2)
{
    return (pair_1.first < pair_2.first);
}


template <typename T>
void compute_ranks(std::vector<int>& ranks, const std::vector<T>& scores)
{
    int nb_observations = (int)scores.size();
    std::vector<std::pair<T,int>> extended_scores(nb_observations);
    for (int k=0; k<nb_observations; ++k)
        extended_scores[k] = std::pair<T,int>(scores[k],k);
    
    // Sort the extended vector, thereby keeping track of the original indices
    std::sort(extended_scores.begin(),extended_scores.end(),compare_pair_wrt_first<T>);
    
    ranks.resize(nb_observations);
    for (int k=0; k<nb_observations; ++k)
        ranks[extended_scores[k].second] = k;
}


template <int d>
void compute_true_transformation_scores(std::vector<double>& transformation_scores, const std::vector<std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>>& transformations_to_evaluate, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_set, EvaluationMeasure eval_measure)
{
    int nb_transformations = (int)transformations_to_evaluate.size();
    transformation_scores = std::vector<double>(nb_transformations,0);
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_locations;
    for (it_locations=target_set.begin(); it_locations!=target_set.end(); ++it_locations)
    {
        Eigen::Matrix<double,d,1> true_value = true_transformation.at(*it_locations); 
//         try
//         {
//             
//         }
//         catch (std::exception e)
//         {
//             std::cout << e.what() << std::endl;
//             Eigen::Matrix<double,d,1> closest_point;
//             std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> input_set;
//             get_input_set(input_set,true_transformation);
//             double dist = get_distance_to_set(closest_point,*it_locations,input_set);
//             std::cout << it_locations->transpose() << closest_point.transpose() << std::endl;
//         }
        for (int t=0; t<nb_transformations; ++t)
        {

            Eigen::Matrix<double,d,1> transformation_value = transformations_to_evaluate[t].at(*it_locations);
            Eigen::Matrix<double,d,1> diff_point = transformation_value - true_value;
          //  std::cout << "True value: " << true_value.transpose() << " - Registered value: " << transformation_value.transpose() << std::endl;
            double diff_norm = diff_point.norm();
            if (eval_measure==l1)
                transformation_scores[t] += diff_norm;
            if (eval_measure==l2)
                transformation_scores[t] += std::pow<double>(diff_norm,2);
            if (eval_measure==l_inf)
                transformation_scores[t] = std::max<double>(transformation_scores[t],diff_norm);
        }
    }
    
    int nb_locations = target_set.size();
    if (eval_measure==l1)
    {
        for (int t=0; t<nb_transformations; ++t)
            transformation_scores[t] = transformation_scores[t]/((double)nb_locations);
    }
    if (eval_measure==l2)
    {
        for (int t=0; t<nb_transformations; ++t)
        {
            transformation_scores[t] = transformation_scores[t]/((double)nb_locations);
            transformation_scores[t] = std::sqrt(transformation_scores[t]);
        }
    }
}


inline void compute_ranking_measures(RankingMeasures& ranking_measures, const std::vector<double>& predicted_raw_scores, const std::vector<double>& true_raw_scores)
{
    int nb_observations = (int)predicted_raw_scores.size();
    
    std::vector<int> predicted_ranks, true_ranks;
    compute_ranks(predicted_ranks,predicted_raw_scores);
    compute_ranks(true_ranks,true_raw_scores);
    
    int ssd_rank(0);
    for (int k=0; k<nb_observations; ++k)
    {
        ssd_rank += std::pow<int>(predicted_ranks[k] - true_ranks[k],2);
        if (predicted_ranks[k]==0) // if this is the transformation ranked as best
            ranking_measures.rank_best_predicted_registration = true_ranks[k] + 1;
        if (true_ranks[k]==0) // if this is the actual best transformation   
            ranking_measures.rank_best_true_registration = predicted_ranks[k] + 1;
    }
    
    ranking_measures.spearman_correlation_coefficient = 1 - 6*((double)ssd_rank)/(((double)nb_observations)*(std::pow<double>((double)nb_observations,2)-1));
    
    double mean_predicted_score(0), mean_true_score(0);
    for (int k=0; k<nb_observations; ++k)
    {
        mean_predicted_score += predicted_raw_scores[k];
        mean_true_score += true_raw_scores[k];
    }
    mean_predicted_score = mean_predicted_score/((double)nb_observations);
    mean_true_score = mean_true_score/((double)nb_observations);
    
    double var_predicted_score(0), var_true_score(0), covariance(0);
    for (int k=0; k<nb_observations; ++k)
    {
        covariance += (predicted_raw_scores[k] - mean_predicted_score)*(true_raw_scores[k] - mean_true_score);
        var_predicted_score += std::pow<double>(predicted_raw_scores[k] - mean_predicted_score,2);
        var_true_score += std::pow<double>(true_raw_scores[k] - mean_true_score,2);
    }
    
    ranking_measures.pearson_correlation_coefficient = covariance/(std::sqrt(var_predicted_score*var_true_score));
    
    double mean_abs_error(0);
    for (int k=0; k<nb_observations; ++k)
        mean_abs_error += std::abs(predicted_raw_scores[k] - true_raw_scores[k]);
    mean_abs_error = mean_abs_error/((double)nb_observations);
    ranking_measures.mean_abs_error_on_score = mean_abs_error;
}

template <int d>
void evaluate_capacity_for_evaluation(std::ofstream& infile, std::ofstream& infile_scores, Gaussian_Process<d>& gp, const std::vector<std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>>& transformations_to_evaluate, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& provided_annotations, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotated_points, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_set, int ind_precomputed_target_set)
{
    EvaluationMeasure eval_measures[3] = {l1, l2, l_inf};
    
    for (int em=0; em<=2; ++em)
    {
        EvaluationMeasure current_eval_measure = eval_measures[em];
        
        // Compute the true, hidden transformation scores
 //       std::cout << "True score" << std::endl;
        std::vector<double> true_transformation_scores;
        compute_true_transformation_scores<d>(true_transformation_scores,transformations_to_evaluate,true_transformation,target_set,current_eval_measure);
//         for (auto s : true_transformation_scores)
//             std::cout << s << " ";
//         std::cout << std::endl;
        
        // Classical landmark-based evaluation: we evaluate at the annotated locations
      //   std::cout << "Classical landmarks" << std::endl;
        std::vector<double> classical_transformation_scores;
        RankingMeasures ranking_measures;
        compute_true_transformation_scores<d>(classical_transformation_scores,transformations_to_evaluate,provided_annotations,annotated_points,current_eval_measure);
        compute_ranking_measures(ranking_measures,classical_transformation_scores,true_transformation_scores);
        infile << ranking_measures.spearman_correlation_coefficient << " " << ranking_measures.pearson_correlation_coefficient <<  " " << ranking_measures.mean_abs_error_on_score << " |";
        
        // The two GP ones 
       //  std::cout << "GP" << std::endl;
        std::vector<std::vector<double>> gp_transformation_scores;
        gp.evaluate_transformations_on_predefined_set(gp_transformation_scores,transformations_to_evaluate,ind_precomputed_target_set,current_eval_measure);
        for (int m=0; m<(gp_transformation_scores.size()); ++m)
        {
            compute_ranking_measures(ranking_measures,gp_transformation_scores[m],true_transformation_scores);
            infile << " " << ranking_measures.spearman_correlation_coefficient << " " << ranking_measures.pearson_correlation_coefficient <<  " " << ranking_measures.mean_abs_error_on_score << " |";
        }
        
        infile << "| ";
        
        if (em==1)
        {
            for (double s : true_transformation_scores)
                 infile_scores << s << " ";
            infile_scores << std::endl;
            for (double s : classical_transformation_scores)
                 infile_scores << s << " ";
            infile_scores << std::endl;
            for (double s : gp_transformation_scores[0])
                 infile_scores << s << " ";
            infile_scores << std::endl;
        }
    }
    infile << std::endl;
}


template <int d>
void heuristic_sampling_for_burn_in(std::vector<Eigen::Matrix<double,d,1>>& output_vector, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& input_full_set, int size_vector)
{
    if (size_vector>input_full_set.size())
    {
        std::cout << "In heuristic_sampling_for_burn_in: Try to sample more points than available in the set" << std::endl;
        size_vector=input_full_set.size();
    }
    
    output_vector.resize(size_vector);
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> selected_points;
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it;
    Eigen::Matrix<double,d,1> best_point;
    for (int k=0; k<size_vector; ++k)
    {
        double best_score(-1);
        for (it=input_full_set.begin(); it!=input_full_set.end(); ++it)
        {
            // Compute score
            double score;
            if (k==0) 
                score = std::rand();
            else
                score = get_distance_to_set(*it,selected_points);
            
            if (score>best_score)
            {
                best_score = score;
                best_point = *it;
            }
        }
        
        output_vector[k] = best_point;
        if (k!=(size_vector-1))
            selected_points.insert(best_point);
    }
}


template <int d>
void heuristic_sampling_for_burn_in(std::vector<Gaussian_Process_Observation<d>>& output_vector, const std::vector<Gaussian_Process_Observation<d>>& input_vector, int size_vector)
{
    if (size_vector>input_vector.size())
    {
        std::cout << "In heuristic_sampling_for_burn_in: Try to sample more points than available in the input vector"  << std::endl;
        size_vector=input_vector.size();
    }
    
    output_vector.resize(size_vector);
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> selected_points;
    Gaussian_Process_Observation<d> best_point;
    for (int k=0; k<size_vector; ++k)
    {
        double best_score(-1);
        for (auto it=input_vector.begin(); it!=input_vector.end(); ++it)
        {
            // Compute score
            double score;
            if (k==0) 
                score = std::rand();
            else
                score = get_distance_to_set(it->input_point,selected_points);
            
            if (score>best_score)
            {
                best_score = score;
                best_point = *it;
            }
        }
        
        output_vector[k] = best_point;
        if (k!=(size_vector-1))
            selected_points.insert(best_point.input_point);
    }
}

template <int d>
void evaluate_query_method(std::vector<Gaussian_Process_Observation<d>>& annotated_observations, const IOSettings& io_settings, const GaussianProcessPrior<d>& gp_prior, const InputLocations<d>& input_locations, const Settings& settings, const InputTransformations<d>& input_transformations, boost::random::mt19937& rng, const QueryingStrategy& querying_strategy, LandmarkAnnotator<d>& landmark_annotator, VisualisationHelper<d>& visualisation_helper)
{
    int nb_burn_in_iterations = (int)input_locations.queries_for_burn_in.size(); 
    bool create_dense_maps = visualisation_helper.create_dense_maps();
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> image_domain_for_display = visualisation_helper.image_domain_for_display();
    
    
    std::string method_name = querying_strategy.get_str();
    int adjusted_nb_iterations = std::min<int>(input_locations.candidate_locations.size(),settings.annotation_budget);
    
    // Identity matrix
    Eigen::Matrix<double,d,d> identity_matrix = Eigen::Matrix<double,d,d>::Identity();
    
    // Create the parameters to estimate 
    MeanFunction<d> *new_mean_function = gp_prior.mean->new_clone();
    KernelFunction<d> *new_kernel_function = gp_prior.kernel->new_clone();
    
    // If there are prior observations, fit them here
    if (gp_prior.prior_observations.size()>0)
    {
        std::cout << "Start learning hyperparameters" << std::endl;
        estimateKernelHyperparametersLOO<d>(*new_kernel_function,gp_prior.prior_observations,*new_mean_function);
        std::cout << "Done" << std::endl;
    }

    // Container for the predicted transformation
    std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>> predicted_transformation, provided_annotations;

    // Create the set of locations in the image domain
    annotated_observations.clear();
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> complement_queried_locations, queried_locations, certain_locations;
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> candidate_locations = input_locations.candidate_locations;
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> target_locations = input_locations.target_locations;
    
    // Create the Gaussian processes
    Gaussian_Process<d> gp_fast_on_X(*new_mean_function,*new_kernel_function), gp_fast_on_T_and_X(*new_mean_function,*new_kernel_function);
    std::vector<Gaussian_Process_Observation<d>> known_values_X, known_values_T_and_X, known_certain_locations;
    
    int nb_precomputed_sets;
    if (create_dense_maps)
        nb_precomputed_sets = 4;
    else
        nb_precomputed_sets = 3;
    std::vector<std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>> precomputed_sets(nb_precomputed_sets);
    precomputed_sets[0] = input_locations.candidate_locations;
    precomputed_sets[1] = input_locations.target_locations;
    precomputed_sets[2] = input_locations.evaluation_locations;
    if (create_dense_maps)
        precomputed_sets[3] = image_domain_for_display;
    gp_fast_on_X.set_predefined_candidate_locations(precomputed_sets);
    
    // If S in entirely contained in T, we will not need the Gaussian process conditioned on T. Else, do the conditioning
    bool are_candidate_locations_subset_of_target = is_A_in_B<Eigen::Matrix<double,d,1>,comp_Point<d>>(candidate_locations,target_locations);
    
    // Condition the Gaussian process on T, if needed
    if (are_candidate_locations_subset_of_target==false)
    {
        typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it_target;
        for (it_target=target_locations.begin(); it_target!=target_locations.end(); ++it_target)
        {
            Gaussian_Process_Observation<d> obs;
            obs.input_point = *it_target;
            obs.output_point = *it_target; // dummy
            obs.observation_noise_covariance_matrix = Eigen::Matrix<double,d,d>::Zero();
            known_values_T_and_X.push_back(obs);
        }
       // 
       
       if (create_dense_maps)
       {
            std::vector<std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>> precomputed_sets_T_on_X(2);
            precomputed_sets_T_on_X[0] = input_locations.candidate_locations;
            precomputed_sets_T_on_X[1] = image_domain_for_display;
            gp_fast_on_T_and_X.set_predefined_candidate_locations(precomputed_sets_T_on_X);
       }
       else
            gp_fast_on_T_and_X.set_predefined_candidate_locations(input_locations.candidate_locations);
    }
    else
        std::cout << "The candidates are included in the target set : we do simpler computations" << std::endl;
    

    
    // Preallocate matrix for entropy
    int nb_target_locations = target_locations.size();

    
    // Helper for entropies
    Helper_Entropy_Decrease_Computation helper;
    
    bool available_first_estimate_hyperparameters = false;
    if (querying_strategy.m_reestimate_variances==false)
        available_first_estimate_hyperparameters = true;
    
    if (available_first_estimate_hyperparameters)
        gp_fast_on_T_and_X.setKnownValues(known_values_T_and_X);
    
    bool exact_computation(false); // for debugging
    
    // Text files for storing results
    std::string outputPrefix = io_settings.results_folder + "/evaluation-measures/" + io_settings.pair_identifier + "-" + io_settings.experiment_identifier + "-" + method_name;
    std::ofstream infile_gp_quality(outputPrefix + "-MeanTransformation.txt");
    std::ofstream infile_evaluation_capacity(outputPrefix + "-EvaluationCapacity.txt");
    std::ofstream infile_transformation_scores(outputPrefix + "-TransformationScores.txt");
    
    // Interaction loop
    for (int l=0; l<=adjusted_nb_iterations; ++l)
    {

        std::cout << "Iteration " << l+1 << "/" << adjusted_nb_iterations + 1 << std::endl;
        std::ostringstream oss_l;
        oss_l << l;
            
        if (l>0) // these are only needed if we have at least one queried location to condition on
        {
                
            //std::cout << "Evaluate quality" << std::endl;
            std::vector<Displacement_Measures> displacement_measures(1); // vector where the evaluation results are stored
            gp_fast_on_X.evaluate_quality_on_predefined_set(infile_gp_quality,predicted_transformation,input_transformations.true_transformation,2);
        
            // Evaluate on all available evaluation locations
            //std::cout << "Evaluate capacity for evaluation" << std::endl;
            evaluate_capacity_for_evaluation<d>(infile_evaluation_capacity,infile_transformation_scores,gp_fast_on_X,input_transformations.transformations_to_evaluate,input_transformations.true_transformation,provided_annotations,queried_locations,input_locations.evaluation_locations,2);
                
            // Visualisation
            std::string window_name = "Predicted transformation";
            visualisation_helper.plot_transformation(window_name,predicted_transformation,input_transformations.true_transformation,queried_locations,input_locations.evaluation_locations);

        }

        if (l==adjusted_nb_iterations)
            break;
            
        // Determine what querying strategy will be used for this iteration
        UncertaintyMeasure uncertainty_measure;
        if (querying_strategy.m_uncertainty_measure == random_placement)
            uncertainty_measure = random_placement;
        if (querying_strategy.m_uncertainty_measure == heuristic)
        {
            if (l==0)
                uncertainty_measure = random_placement;
            else
                uncertainty_measure = heuristic;
        }
        if (querying_strategy.m_uncertainty_measure == entropy)
        {
            if (l==0)
                uncertainty_measure = random_placement;  
            else
            {
                if (available_first_estimate_hyperparameters == false) // if we cannot use GPs yet
                    uncertainty_measure = heuristic;
                else
                    uncertainty_measure = entropy;
            }
        }
            
        // Test whether the set of candidate locations is fully included
        if (!are_candidate_locations_subset_of_target)
            are_candidate_locations_subset_of_target = is_A_in_B<Eigen::Matrix<double,d,1>,comp_Point<d>>(candidate_locations,target_locations);
            

        // Iterate over the (remaining) candidate locations
        //std::cout << "Iterations over candidates" << std::endl;
        double best_score(-1);
        double heuristic_score_best_candidate(-1);
        Eigen::Matrix<double,d,1> best_candidate;
            
        if (l<nb_burn_in_iterations)
            best_candidate = input_locations.queries_for_burn_in[l];
        else
        {

            // Precompute entropies and the determinant if fast computation
            std::map<Eigen::Matrix<double,d,1>,double,comp_Point<d>> entropies_on_X, entropies_on_T_and_X;
            double current_det(0);
            if (uncertainty_measure==entropy)
            {
                gp_fast_on_X.compute_entropies_of_predefined_candidates(entropies_on_X,0);
                if (are_candidate_locations_subset_of_target==false)
                    gp_fast_on_T_and_X.compute_entropies_of_predefined_candidates(entropies_on_T_and_X,0);
            }

                            
            for (auto it_candidate=candidate_locations.begin(); it_candidate!=candidate_locations.end(); ++it_candidate)
            {
                Eigen::Matrix<double,d,1> current_x = *it_candidate;
                double final_score;
                    
                if (uncertainty_measure == random_placement) 
                    final_score = std::rand();

                if (uncertainty_measure == heuristic)
                    final_score = get_distance_to_set(current_x,queried_locations);
                    
                if (uncertainty_measure == entropy)
                {
                    if (target_locations.count(current_x)) // if x in T
                        final_score = entropies_on_X[current_x];
                    else
                        final_score = entropies_on_X[current_x] - entropies_on_T_and_X[current_x];     
                } 
                       
                if ((final_score>best_score) || (it_candidate==candidate_locations.begin()))
                {
                    best_score = final_score;
                    best_candidate = current_x;
                    heuristic_score_best_candidate = get_distance_to_set(current_x,queried_locations);
                }
                else
                {
                    if (final_score==best_score) 
                    {
                        std::cout << "There is a tie!" << std::endl;
                        double current_heuristic_score = get_distance_to_set(current_x,queried_locations);
                        if (current_heuristic_score>heuristic_score_best_candidate)
                        {
                            best_candidate = current_x;
                            heuristic_score_best_candidate = current_heuristic_score;
                        }
                    }
                }
                    

            }
            
        }
        
            
        // Queried location
        queried_locations.insert(best_candidate);

        // We collect the user label
        bool no_point_provided(false);
        bool is_observation_uncertain(true);
        Gaussian_Process_Observation<d> obs;
        obs.input_point = best_candidate;
        Eigen::Matrix<double,d,1> predicted_location;
        gp_fast_on_X.mean(predicted_location,obs.input_point);
        landmark_annotator.getAnnotation(obs,is_observation_uncertain,no_point_provided,best_candidate,predicted_location);
            
        if (!no_point_provided)
        {
            
            provided_annotations[obs.input_point] = obs.output_point;
            annotated_observations.push_back(obs);
            known_values_X.push_back(obs);
            known_values_T_and_X.push_back(obs);
            
            
            // We delete the chosen x from the candidate locations
            candidate_locations.erase(best_candidate);
            
            // Update GPs
            gp_fast_on_X.addKnownValue(obs);
            if (are_candidate_locations_subset_of_target==false)
                gp_fast_on_T_and_X.addKnownValue(obs);
        }
                
        // Reestimate GP
        if (((querying_strategy.m_reestimate_variances) && ((l + 1) == nb_burn_in_iterations)) || (create_dense_maps) && ((l + 1) == adjusted_nb_iterations))
        {
            std::cout << "Reestimate mean" << std::endl;
            new_mean_function->fit(known_values_X);

            std::cout << "Reestimate kernel parameters" << std::endl;
            if (gp_prior.prior_observations.size()>0)
            {
                std::vector<std::vector<Gaussian_Process_Observation<d>>> all_observations = gp_prior.prior_observations;
                all_observations.push_back(known_values_X);
                estimateKernelHyperparametersLOO<d>(*new_kernel_function,all_observations,*new_mean_function);
            }
            else
                estimateKernelHyperparametersLOO<d>(*new_kernel_function,known_values_X,*new_mean_function);
            
            std::cout << "Update Gaussian processes" << std::endl;
            gp_fast_on_X.reset(*new_mean_function,*new_kernel_function,known_values_X);
            if (are_candidate_locations_subset_of_target==false)
                gp_fast_on_T_and_X.reset(*new_mean_function,*new_kernel_function,known_values_T_and_X);
                    
            available_first_estimate_hyperparameters = true;
        }
            

        if (no_point_provided)
            break;

    }
     
    if (create_dense_maps)
    {
        int nb_transformations_to_evaluate = (int)input_transformations.transformations_to_evaluate.size();
        for (int t=0; t<nb_transformations_to_evaluate; ++t)
        {
            std::ostringstream oss_t;
            oss_t << t;
            
            visualisation_helper.plot_heat_map("Joint heat map", io_settings.results_folder + "/joint-heat-maps/",io_settings.pair_identifier + "-" + io_settings.experiment_identifier + "-" + method_name + "-EP" + oss_t.str(), input_transformations.transformations_to_evaluate[t], 3, gp_fast_on_X);
       //     plot_heat_map_outliers_at_predefined_candidates(gp_fast_on_X,transformations_to_evaluate[t],3,fixed_image.rows,fixed_image.cols,"Heat map outliers",subsampling_display,results_folder + "/HeatMapsOutliers",pair_identifier + "-" + experiment_identifier + "-" + method_name + "-GS" + oss_t.str());
        }
    }
     
    delete new_mean_function;
    delete new_kernel_function;
}


#endif
