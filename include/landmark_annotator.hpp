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


#ifndef LANDMARK_ANNOTATOR_H_INCLUDED
#define LANDMARK_ANNOTATOR_H_INCLUDED

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "gp_observation_database.hpp"



struct comp_Point2f {
  bool operator() (const cv::Point2f& p_1, const cv::Point2f& p_2) const
  {return (((p_1.y)<(p_2.y)) || (((p_1.y)==(p_2.y)) && ((p_1.x)<(p_2.x))));}
};

template <int d>
class LandmarkAnnotator
{
	public:
	virtual ~LandmarkAnnotator() = 0;
    virtual void getAnnotation(Gaussian_Process_Observation<d>& obs, bool& is_observation_uncertain, bool& no_point_provided, const Eigen::Matrix<double,d,1>& queried_location, const Eigen::Matrix<double,d,1>& predicted_location) = 0;
};


template <int d>
LandmarkAnnotator<d>::~LandmarkAnnotator<d>()
{}


template <int d>
void draw_sample_from_diagonalised_covariance_matrix(Eigen::Matrix<double,d,1>& sample, const Eigen::Matrix<double,d,d>& D, const Eigen::Matrix<double,d,d>& V, boost::random::mt19937& rng)
{
    boost::normal_distribution<double> normal_dist(0.0,1.0);
    Eigen::Matrix<double,d,1> axis_aligned_sample;
    for (int i=0; i<d; ++i)
        axis_aligned_sample(i) = std::sqrt(D(i,i))*normal_dist(rng);
    sample = V*axis_aligned_sample;
}


template <int d>
void draw_sample_from_diagonal_covariance_matrix(Eigen::Matrix<double,d,1>& sample, const Eigen::Matrix<double,d,d>& D, boost::random::mt19937& rng)
{
    Eigen::Matrix<double,d,d> I;
    I.setIdentity();
    draw_sample_from_diagonalised_covariance_matrix<d>(sample,D,I,rng);
}


template <int d>
void draw_random_orthogonal_matrix(Eigen::Matrix<double,d,d>& V, boost::random::mt19937& rng)
{
     static_assert(((d==2) || (d==3)),"draw_random_orthogonal_matrix is only implemented in dimension 2 or 3");
}

template <>
inline void draw_random_orthogonal_matrix(Eigen::Matrix2d& V, boost::random::mt19937& rng)
{
    boost::uniform_real<double> angle_distribution(0,2*M_PI);
    double theta = angle_distribution(rng);
     
    V(0,0) = std::cos(theta);
    V(1,0) = -std::sin(theta);
    V(0,1) = std::sin(theta);
    V(1,1) = std::cos(theta);
}

/*
template <>
inline void draw_random_orthogonal_matrix(Eigen::Matrix3d& V, boost::random::mt19937& rng)
{
    boost::uniform_real<double> angle_distribution(0,2*M_PI);
    double theta_1 = angle_distribution(rng);
    double theta_2 = angle_distribution(rng);
    double theta_3 = angle_distribution(rng);
    
    
    Eigen::Matrix3d V1 = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d V2 = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d V3 = Eigen::Matrix3d::Zero();
    
    // Fill V1
    V1(0,0) = 1;
    V1(1,1) = std::cos(theta_1);
    V1(2,2) = std::cos(theta_1);
    V1(2,1) = std::sin(theta_1);
    V1(1,2) = -std::sin(theta_1);
    
    // Fill V2
    V2(1,1) = 1;
    V2(0,0) = std::cos(theta_2);
    V2(2,2) = std::cos(theta_2);
    V2(0,2) = std::sin(theta_2);
    V2(2,0) = -std::sin(theta_2);
    
    // Fill V3
    V3(2,2) = 1;
    V3(0,0) = std::cos(theta_3);
    V3(1,0) = std::sin(theta_3);
    V3(0,1) = -std::sin(theta_3);
    V3(1,1) = std::cos(theta_3);
    
    V = V1*V2*V3;
}*/

// Adapted from Eigen's UnitRandom() static method in the Quaternion class
// Itself based on: http://planning.cs.uiuc.edu/node198.html 
template <>
inline void draw_random_orthogonal_matrix(Eigen::Matrix3d& V, boost::random::mt19937& rng)
{
    boost::uniform_real<double> angle_distribution(0,2*M_PI);
    boost::uniform_real<double> unif01(0,1);
    
    double u1 = unif01(rng);
    double u2 = angle_distribution(rng);
    double u3 = angle_distribution(rng);
    const double a = std::sqrt(1 - u1);
    const double b = std::sqrt(u1);
    Eigen::Quaterniond q(a * std::sin(u2), a * std::cos(u2), b * std::sin(u3), b * std::cos(u3));
    
    V = q.toRotationMatrix();
}


template <int d>
void draw_random_covariance_matrix(Eigen::Matrix<double,d,d>& covariance_matrix, Eigen::Matrix<double,d,1>& sample, boost::uniform_real<double>& annotation_noise_distribution, boost::random::mt19937& rng, double confidence_level)
{
    double gamma_square = 2*boost::math::gamma_p_inv(d/2,1 - confidence_level);
    
    // Draw the random angle and radius in both directions
    Eigen::Matrix<double,d,d> V, D;
    
    draw_random_orthogonal_matrix<d>(V,rng);
    
    D = Eigen::Matrix<double,d,d>::Zero();
    for (int i=0; i<d; ++i)
    {
        double a_i = annotation_noise_distribution(rng);
        D(i,i) = a_i*a_i/gamma_square;
    }
    
    covariance_matrix = V*D*(V.transpose());
    
    // Create a sample from this covariance matrix
    draw_sample_from_diagonalised_covariance_matrix<d>(sample,D,V,rng);
}


template <int d>
void draw_random_covariance_matrix_lognormal(Eigen::Matrix<double,d,d>& covariance_matrix, Eigen::Matrix<double,d,1>& sample, boost::normal_distribution<double>& normal_distribution_log_eigenvalues, boost::random::mt19937& rng, double confidence_level)
{
    // Draw the random angle and radius in both directions
    Eigen::Matrix<double,d,d> V, D;
    
    draw_random_orthogonal_matrix<d>(V,rng);
    
    D = Eigen::Matrix<double,d,d>::Zero();
    for (int i=0; i<d; ++i)
        D(i,i) = std::exp(normal_distribution_log_eigenvalues(rng));
    
    covariance_matrix = V*D*(V.transpose());
    
    // Create a sample from this covariance matrix
    draw_sample_from_diagonalised_covariance_matrix<d>(sample,D,V,rng);
}

template <int d>
void draw_random_sample_lognormal_multiple_annotators(std::vector<Eigen::Matrix<double,d,1>>& samples, boost::normal_distribution<double>& normal_distribution_log_error, boost::random::mt19937& rng, int nb_annotators)
{
    samples.resize(nb_annotators);
    
    // Draw the random axes and direction along which the positive error is applied
    Eigen::Matrix<double,d,d> V;

    // Get samples from this covariance matrix
    Eigen::Matrix<double,d,1> positive_sample;
    for (int n=0; n<nb_annotators; ++n)
    {
        draw_random_orthogonal_matrix<d>(V,rng);
        for (int k=0; k<d; ++k)
            positive_sample(k) = std::exp(normal_distribution_log_error(rng));
        samples[n] = V*positive_sample;
    }
}


template <int d>
void create_default_covariance_matrix(Eigen::Matrix<double,d,d>& default_covariance_matrix, double default_radius, double confidence_level)
{
    double gamma_square = 2*boost::math::gamma_p_inv(d/2,1 - confidence_level);
    default_covariance_matrix.setZero();
    for (int i=0; i<d; ++i)
        default_covariance_matrix(i,i) = default_radius*default_radius/gamma_square;
}



template <int d>
void draw_simulated_annotations(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>,comp_Point<d>>& annotation_covariance_matrices, std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& annotation_noisy_outputs, std::map<Eigen::Matrix<double,d,1>,bool,comp_Point<d>>& is_annotation_uncertain, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, boost::uniform_real<double>& annotation_noise_distribution, boost::random::mt19937& rng, double confidence_level, double proportion_difficult_locations, const Eigen::Matrix<double,d,d>& default_covariance_matrix)
{
    annotation_covariance_matrices.clear();
    annotation_noisy_outputs.clear();
    is_annotation_uncertain.clear();
    
    boost::uniform_real<double> unif_01(0,1);
    
    Eigen::Matrix<double,d,d> random_cov_matrix;
    Eigen::Matrix<double,d,1> random_sample;
    
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it;
    for (it=annotable_locations.begin(); it!=annotable_locations.end(); ++it)
    {
        if (unif_01(rng)<proportion_difficult_locations) // if the location is a difficult one
        {
            // Draw the mean and covariance of the annotations
            draw_random_covariance_matrix<d>(random_cov_matrix,random_sample,annotation_noise_distribution,rng,confidence_level);
            annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,random_cov_matrix));
            is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,true));
        }
        else
        {
            draw_sample_from_diagonal_covariance_matrix<d>(random_sample,default_covariance_matrix,rng);
            annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,default_covariance_matrix));
            is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,false));
        }
        
        
        Eigen::Matrix<double,d,1> mean_point = true_transformation.at(*it) + random_sample;
        
      // Here, we round the output to simulate the fact that a user is limited by the resolution
      //  for (int i=0; i<d; ++i)
      //      mean_point(i) = std::round(mean_point(i));
        
        annotation_noisy_outputs.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(*it,mean_point));
    }
}

template <int d>
void draw_simulated_annotations_lognormal(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>,comp_Point<d>>& annotation_covariance_matrices, std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& annotation_noisy_outputs, std::map<Eigen::Matrix<double,d,1>,bool,comp_Point<d>>& is_annotation_uncertain, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, boost::normal_distribution<double>& normal_distribution_log_eigenvalues, boost::random::mt19937& rng, double confidence_level, double proportion_difficult_locations, const Eigen::Matrix<double,d,d>& default_covariance_matrix, bool pretend_annotations_are_certain)
{
    annotation_covariance_matrices.clear();
    annotation_noisy_outputs.clear();
    is_annotation_uncertain.clear();
    
    boost::uniform_real<double> unif_01(0,1);
    
    Eigen::Matrix<double,d,d> random_cov_matrix;
    Eigen::Matrix<double,d,1> random_sample;
    
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it;
    for (it=annotable_locations.begin(); it!=annotable_locations.end(); ++it)
    {
        if ((unif_01(rng)<proportion_difficult_locations) || (pretend_annotations_are_certain)) // if the location is a difficult one
        {
            // Draw the mean and covariance of the annotations
            draw_random_covariance_matrix_lognormal<d>(random_cov_matrix,random_sample,normal_distribution_log_eigenvalues,rng,confidence_level);
            
            if (pretend_annotations_are_certain)
            {
                annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,default_covariance_matrix));
                is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,false));
            }
            else
            {
                annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,random_cov_matrix));
                is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,true));
            }
        }
        else
        {
            draw_sample_from_diagonal_covariance_matrix<d>(random_sample,default_covariance_matrix,rng);
            annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,default_covariance_matrix));
            is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,false));
        }
        
        
        Eigen::Matrix<double,d,1> mean_point = true_transformation.at(*it) + random_sample;
        
    // Here, we round the output to simulate the fact that a user is limited by the resolution (not always relevant though if spacing is used)
    //    for (int i=0; i<d; ++i)
    //        mean_point(i) = std::round(mean_point(i));
        
        annotation_noisy_outputs.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(*it,mean_point));
    }
    
    
    
    
    
}


template <int d>
void draw_simulated_annotations_lognormal_multiple_annotators(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>,comp_Point<d>>& annotation_covariance_matrices, std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& annotation_noisy_outputs, std::map<Eigen::Matrix<double,d,1>,bool,comp_Point<d>>& is_annotation_uncertain, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, boost::normal_distribution<double>& normal_distribution_log_eigenvalues, boost::random::mt19937& rng, double confidence_level, double proportion_difficult_locations, const Eigen::Matrix<double,d,d>& default_covariance_matrix, bool pretend_annotations_are_certain, int nb_annotators)
{
    annotation_covariance_matrices.clear();
    annotation_noisy_outputs.clear();
    is_annotation_uncertain.clear();
    
    boost::uniform_real<double> unif_01(0,1);
    
    std::vector<double> annotation_errors;
    

    std::vector<Eigen::Matrix<double,d,1>> random_samples(nb_annotators);
    
    typename std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>::const_iterator it;
    for (it=annotable_locations.begin(); it!=annotable_locations.end(); ++it)
    {
        if (unif_01(rng)<proportion_difficult_locations) // if the location is a difficult one
        {
            // Draw the mean and covariance of the annotations
            draw_random_sample_lognormal_multiple_annotators<d>(random_samples,normal_distribution_log_eigenvalues,rng,nb_annotators);
            is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,true));
        }
        else
        {
            for (int n=0; n<nb_annotators; ++n)
                draw_sample_from_diagonal_covariance_matrix<d>(random_samples[n],default_covariance_matrix,rng);
            is_annotation_uncertain.insert(std::pair<Eigen::Matrix<double,d,1>,bool>(*it,false));
        }
        
        
        // Compute the mean sample
        Eigen::Matrix<double,d,1> mean_sample = Eigen::Matrix<double,d,1>::Zero();
        for (int n=0; n<nb_annotators; ++n)
            mean_sample += random_samples[n];
        mean_sample = (1/((double)nb_annotators))*mean_sample;
        Eigen::Matrix<double,d,1> mean_point = true_transformation.at(*it) + mean_sample;
        
       
        if ((pretend_annotations_are_certain) || (nb_annotators < 3))
            annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,default_covariance_matrix));
        else
        {
            // Compute the sample covariance matrix
            Eigen::Matrix<double,d,d> sample_covariance_matrix = Eigen::Matrix<double,d,d>::Zero();
            for (int n=0; n<nb_annotators; ++n)
            {
                Eigen::Matrix<double,d,1> diff_sample = random_samples[n] - mean_sample;
                sample_covariance_matrix += diff_sample*(diff_sample.transpose());
            }
            sample_covariance_matrix = (1/((double)nb_annotators-1))*sample_covariance_matrix;
            annotation_covariance_matrices.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>>(*it,sample_covariance_matrix));
        }
        
        annotation_noisy_outputs.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(*it,mean_point));
        
        
        double error = random_samples[0].norm();
        annotation_errors.push_back(error);
        
//         std::cout << "True value: " << true_transformation.at(*it).transpose() << std::endl;
//         std::cout << "Sampled value: " << mean_point.transpose() << std::endl;
//         std::cout << "Covariance matrix: " << sample_covariance_matrix << std::endl;
//         std::cout << "----" << std::endl;
        
    }
    
    
    
    int N = annotation_errors.size();
    double mean_error(0), std_error(0);
    for (double e : annotation_errors)
        mean_error += e;
    mean_error = (1/((double)N))*mean_error;
    for (double e : annotation_errors)
        std_error += std::pow<double>(e - mean_error,2);
    std_error = std::sqrt(std_error/((double)N-1));
    
    std::cout << "Sampled annotation distribution: " << mean_error << " " << std_error << std::endl;
    
    
}








// -------------------------------
// Precomputed landmark annotator
// -------------------------------

template <int d>
class PrecomputedLandmarkAnnotator : public LandmarkAnnotator<d>
{
	public:
    void createfromBimodalDistribution(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, double default_radius, double min_radius, double max_radius, double confidence_level, double proportion_difficult_locations, boost::random::mt19937& rng);
    void createfromBimodalLognormalDistribution(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, double default_radius, double confidence_level, double mean_log_eigenvalues, double std_log_eigenvalues, double proportion_difficult_locations, boost::random::mt19937& rng, bool pretend_annotations_are_certain = false, int nb_annotators = 1);
    void getAnnotation(Gaussian_Process_Observation<d>& obs, const Eigen::Matrix<double,d,1>& queried_location, const Eigen::Matrix<double,d,1>& predicted_mean);
    void getAnnotation(Gaussian_Process_Observation<d>& obs, bool& is_observation_uncertain, bool& no_point_provided, const Eigen::Matrix<double,d,1>& queried_location, const Eigen::Matrix<double,d,1>& predicted_location);
    std::vector<Gaussian_Process_Observation<d>> getAllAnnotations();
    
    private:
    std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>> m_annotation_means;
    std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,d>,comp_Point<d>> m_annotation_covariance_matrices;
    std::map<Eigen::Matrix<double,d,1>,bool,comp_Point<d>> m_is_annotation_uncertain;

};


template <int d>
void PrecomputedLandmarkAnnotator<d>::createfromBimodalDistribution(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, double default_radius, double min_radius, double max_radius, double confidence_level, double proportion_difficult_locations, boost::random::mt19937& rng)
{
    // Clear current maps
    m_annotation_means.clear();
    m_annotation_covariance_matrices.clear();
    m_is_annotation_uncertain.clear();
    
    // Create the noise distribution of "difficult samples"
    boost::uniform_real<double> annotation_noise_distribution(min_radius,max_radius); 
    
    // Create the default matrix for "easy" samples
    Eigen::Matrix<double,d,d> default_covariance_matrix;
    create_default_covariance_matrix(default_covariance_matrix,default_radius,confidence_level);
    
    // Draw and store the precomputed annotations
    draw_simulated_annotations<d>(this->m_annotation_covariance_matrices,this->m_annotation_means,this->m_is_annotation_uncertain,annotable_locations,true_transformation,annotation_noise_distribution,rng,confidence_level,proportion_difficult_locations,default_covariance_matrix);
}

template <int d>
void PrecomputedLandmarkAnnotator<d>::createfromBimodalLognormalDistribution(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& annotable_locations, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, double default_radius, double confidence_level, double mean_log_eigenvalues, double std_log_eigenvalues, double proportion_difficult_locations, boost::random::mt19937& rng, bool pretend_annotations_are_certain, int nb_annotators)
{
    // Clear current maps
    m_annotation_means.clear();
    m_annotation_covariance_matrices.clear();
    m_is_annotation_uncertain.clear();
     
    boost::normal_distribution<double> normal_distribution_log_eigenvalues(mean_log_eigenvalues,std_log_eigenvalues);
    
    // Create the default matrix for "easy" samples
    Eigen::Matrix<double,d,d> default_covariance_matrix;
    create_default_covariance_matrix(default_covariance_matrix,default_radius,confidence_level);
     
    if (nb_annotators == 1)
    {
        // Draw and store the precomputed annotations
        draw_simulated_annotations_lognormal<d>(this->m_annotation_covariance_matrices,this->m_annotation_means,this->m_is_annotation_uncertain,annotable_locations,true_transformation,normal_distribution_log_eigenvalues,rng,confidence_level,proportion_difficult_locations,default_covariance_matrix,pretend_annotations_are_certain);
    }
    else
    {
         draw_simulated_annotations_lognormal_multiple_annotators<d>(this->m_annotation_covariance_matrices,this->m_annotation_means,this->m_is_annotation_uncertain,annotable_locations,true_transformation,normal_distribution_log_eigenvalues,rng,confidence_level,proportion_difficult_locations,default_covariance_matrix,pretend_annotations_are_certain,nb_annotators);
    }
}

template <int d>
void PrecomputedLandmarkAnnotator<d>::getAnnotation(Gaussian_Process_Observation<d>& obs, bool& is_observation_uncertain, bool& no_point_provided, const Eigen::Matrix<double,d,1>& queried_location, const Eigen::Matrix<double,d,1>& predicted_location)
{
    obs.input_point = queried_location;
    if ((this->m_annotation_means.count(queried_location)==0) || (this->m_annotation_covariance_matrices.count(queried_location)==0))
    {
        std::cout << "Queried point was not found in the precomputed annotations" << std::endl;
        no_point_provided = true;
    }
    else
    {
        obs.output_point = this->m_annotation_means.at(queried_location);
        obs.observation_noise_covariance_matrix = this->m_annotation_covariance_matrices.at(queried_location);
        is_observation_uncertain = this->m_is_annotation_uncertain.at(queried_location);
        no_point_provided = false;
    }
}


template <int d>
std::vector<Gaussian_Process_Observation<d>> PrecomputedLandmarkAnnotator<d>::getAllAnnotations()
{
    std::vector<Gaussian_Process_Observation<d>> res;
    for (auto it : m_annotation_means)
    {
        Gaussian_Process_Observation<d> obs;
        Eigen::Matrix<double,d,1> queried_location = it.first;
        bool is_observation_uncertain, no_point_provided;
        this->getAnnotation(obs,is_observation_uncertain, no_point_provided, queried_location, queried_location);
        res.push_back(obs);
    }
    return res;
}

#endif
