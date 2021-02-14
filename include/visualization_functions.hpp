#ifndef VISUALIZATION_FUNCTIONS_INCLUDED
#define VISUALIZATION_FUNCTIONS_INCLUDED

#include <vector>
#include <map>
#include <set>
#include <Eigen/Dense>
#include "gp_observation_database.hpp"
#include "gaussian_process.hpp"
#include "processing_scripts.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>
#include <boost/math/special_functions/gamma.hpp>


template <int d>
class VisualisationHelper
{
public:
    VisualisationHelper(const Eigen::Matrix<double,d,1>& spacing = Eigen::Matrix<double,d,1>::Zero()) : m_create_dense_maps(false), m_spacing(spacing) {};
    virtual ~VisualisationHelper() = default; 
    inline std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> image_domain_for_display() const {return m_image_domain_for_display;};
    inline bool create_dense_maps() const {return m_create_dense_maps;};
    virtual void plot_transformation(const std::string& window_name, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& predicted_transformation, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& queried_locations, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& evaluation_locations) {};
    virtual void plot_heat_map(const std::string& window_name, const std::string& heat_map_folder, const std::string pair_identifier, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& transformation_to_evaluate, int ind_predefined_target_set, Gaussian_Process<d>& gp_on_X) {};
    
protected:
    std::set<Eigen::Matrix<double,d,1>,comp_Point<d>> m_image_domain_for_display;
    bool m_create_dense_maps;
    Eigen::Matrix<double,d,1> m_spacing;
};
/*
template <int d>
void VisualisationHelper<d>::plot_transformation(const std::string& window_name, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& predicted_transformation, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& queried_locations, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& evaluation_locations)
{}*/



template <>
class VisualisationHelper<2>
{
public:
    VisualisationHelper(const cv::Mat& fixed_image, const cv::Mat& moving_image, int subsampling_dense_maps, bool create_dense_maps, const Eigen::Vector2d& spacing, bool display = true);
    ~VisualisationHelper() = default;
    inline bool create_dense_maps() const {return m_create_dense_maps;};
    inline std::set<Eigen::Vector2d,comp_Point<2>> image_domain_for_display() const {return m_image_domain_for_display;};
    void plot_transformation(const std::string& window_name, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& predicted_transformation, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& true_transformation, const std::set<Eigen::Vector2d,comp_Point<2>>& queried_locations, const std::set<Eigen::Vector2d,comp_Point<2>>& evaluation_locations);
    void plot_heat_map(const std::string& window_name, const std::string& heat_map_folder, const std::string pair_identifier, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, Gaussian_Process<2>& gp_on_X);
private:
    std::set<Eigen::Vector2d,comp_Point<2>> m_image_domain_for_display;
    bool m_create_dense_maps;
    Eigen::Vector2d m_spacing;
    cv::Mat m_fixed_image;
    cv::Mat m_fixed_image_for_display;
    cv::Mat m_moving_image;
    cv::Mat m_entropy_display;
    int m_subsampling_dense_maps;
    bool m_display;
};


template <int d>
double compute_outlier_score_for_evaluation(const Eigen::Matrix<double,d,1>& predicted_location, const Eigen::Matrix<double,d,1>& mean_true_location, const Eigen::Matrix<double,d,d>& covariance_true_location)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,d,d>> es;
    es.computeDirect(covariance_true_location);
    Eigen::Matrix2d V = es.eigenvectors();
    Eigen::Matrix<double,d,1> s = es.eigenvalues();
    Eigen::Matrix<double,d,1> dot_products = (V.transpose())*(predicted_location - mean_true_location);
    double gamma_square(0);
    for (int k=0; k<d; ++k)
        gamma_square += std::pow<double>(dot_products(k)/s(k),2);
    return boost::math::gamma_p(0.5*((double)d),0.5*gamma_square);
}

cv::Point2f eigen_to_cv(const Eigen::Vector2d& eigen_point);

void plot_landmarks_on_image(cv::Mat& image, const std::set<Eigen::Vector2d,comp_Point<2>>& landmarks, int radius, const cv::Scalar& color, int thickness);

void plot_landmarks_on_image(cv::Mat& image, const std::set<Eigen::Vector2d,comp_Point<2>>& landmarks, int markerType, int markerSize, const cv::Scalar& color, int thickness);

void normalise_and_write(const cv::Mat& input_image, const std::string& filename);

void plot_salient_and_target_points(const cv::Mat& input_image, const std::set<Eigen::Vector2d,comp_Point<2>>& salient_points, const std::set<Eigen::Vector2d,comp_Point<2>>& target_points, const std::string& filename);

void plot_2D_transformation(const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, const std::string& window_name, const cv::Scalar& color, int nb_rows, int nb_cols);
// Plots a field
void plot_2D_transformation(cv::Mat& output_image, const cv::Mat& background_image, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, const std::string& window_name, const cv::Scalar& color);

void plot_2D_transformation(cv::Mat& output_image, const cv::Mat& background_image, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, const std::set<Eigen::Vector2d,comp_Point<2>>& locations, const std::string& window_name, const cv::Scalar& color);

void create_heat_map_outliers(cv::Mat& heat_map, Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, const std::set<Eigen::Vector2d,comp_Point<2>>& target_set, int nb_rows, int nb_cols);

void plot_heat_map_outliers(Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, const std::set<Eigen::Vector2d,comp_Point<2>>& target_set, int nb_rows, int nb_cols, const std::string& window_name);

void create_heat_map_outliers_at_predefined_candidates(cv::Mat& heat_map, Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, int nb_rows, int nb_cols);

void create_heat_map_entropy_at_predefined_candidates(cv::Mat& heat_map, Gaussian_Process<2>& gp, int ind_predefined_target_set, int nb_rows, int nb_cols);

void plot_heat_map_outliers_at_predefined_candidates(Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, int nb_rows, int nb_cols, const std::string& window_name, int subsampling_display, const std::string& heat_map_folder, const std::string pair_identifier);

void plot_heat_map_entropy_at_predefined_candidates(Gaussian_Process<2>& gp, int ind_predefined_target_set, int nb_rows, int nb_cols, const std::string& window_name, int subsampling_display, const std::string& heat_map_folder, const std::string pair_identifier);

void plot_joint_heat_map_at_predefined_candidates(Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, int nb_rows, int nb_cols, const cv::Mat fixed_image, const std::string& window_name, int subsampling_display, const std::string& joint_map_folder, const std::string pair_identifier);

void plot_heat_map_entropy_at_predefined_candidates(Gaussian_Process<2>& gp_on_X, Gaussian_Process<2>& gp_on_T_and_X, int ind_predefined_target_set_on_X, int ind_predefined_target_set_T_on_X, int nb_rows, int nb_cols, const std::string& window_name, int subsampling_display, const std::string& heat_map_folder, const std::string pair_identifier);


#endif // REGISTRATION_AVERAGING_INCLUDED
