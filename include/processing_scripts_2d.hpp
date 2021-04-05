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

#ifndef PROCESSING_SCRIPTS_2D_INCLUDED
#define PROCESSING_SCRIPTS_2D_INCLUDED

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <ctime>

#include <omp.h>

#include "landmark_suggestion.hpp"
#include "mean_function.hpp"
#include "gaussian_kernel.hpp"
#include "wendland_kernel.hpp"
#include "gaussian_process.hpp"
#include "processing_scripts.hpp"

// For solving system
#include "stdafx.h"
#include "optimization.h"


#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/xfeatures2d.hpp>

// Random number / probabilities / sampling
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>




// This function assumes we are on a pixel grid (ie x2 = x1 + 1 and y2 = y1 + 1) , dx = x - x1
template <typename PixelType>
void bilinear_2D_interpolation(PixelType& val_u, PixelType& val_v, PixelType dx, PixelType dy, PixelType f_11_u, PixelType f_21_u, PixelType f_12_u, PixelType f_22_u, PixelType f_11_v, PixelType f_21_v, PixelType f_12_v, PixelType f_22_v)   // fu(x1,y1) , fu(x2,y1), fu(x1,y2), fu(x2,y2), fv(x1,y1) , fv(x2,y1), fv(x1,y2), fv(x2,y2)
{
    // Compute weights
    PixelType w_22 = dx*dy;
    PixelType w_21 = dx - w_22;
    PixelType w_12 = dy - w_22;
    PixelType w_11 = 1 - dx - w_12;

    // Output value
    val_u = w_11*f_11_u + w_21*f_21_u + w_12*f_12_u + w_22*f_22_u;
    val_v = w_11*f_11_v + w_21*f_21_v + w_12*f_12_v + w_22*f_22_v;
}


template <typename PixelType>
void square_exponential_vector_field(cv::Mat& output_map, cv::Mat& input_map)
{
    int dimx = input_map.cols;
    int dimy = input_map.rows;
    for (int y=0; y<dimy; ++y)
    {
        PixelType *input_ptr = input_map.ptr<PixelType>(y);
        PixelType *output_ptr = output_map.ptr<PixelType>(y);
        for (int x=0; x<dimx; ++x)
        {
            // Target location at which we have to interpolate
            PixelType target_x = input_ptr[2*x];
            PixelType target_y = input_ptr[2*x+1];

            if ((target_x<0) || (target_x>(dimx-1)) || (target_y<0) || (target_y>(dimy-1)))
            {
                output_ptr[2*x] = target_x;
                output_ptr[2*x+1] = target_y;
            }
            else
            {
                int target_x_low = (int)floor(target_x);
                int target_y_low = (int)floor(target_y);
                if (target_x_low==(dimx-1))
                    --target_x_low;
                if (target_y_low==(dimy-1))
                    --target_y_low;
                int target_x_up = target_x_low + 1;
                int target_y_up = target_y_low + 1;
                PixelType dx = target_x - (PixelType)target_x_low;
                PixelType dy = target_y - (PixelType)target_y_low;
                PixelType *input_target_y_low_ptr = input_map.ptr<PixelType>(target_y_low);
                PixelType *input_target_y_up_ptr = input_map.ptr<PixelType>(target_y_up);
                bilinear_2D_interpolation<PixelType>(output_ptr[2*x],output_ptr[2*x+1],dx,dy,input_target_y_low_ptr[2*target_x_low],input_target_y_low_ptr[2*target_x_up],input_target_y_up_ptr[2*target_x_low],input_target_y_up_ptr[2*target_x_up],input_target_y_low_ptr[2*target_x_low+1],input_target_y_low_ptr[2*target_x_up+1],input_target_y_up_ptr[2*target_x_low+1],input_target_y_up_ptr[2*target_x_up+1]);
            }
        }
    }

}


void performScalingAndSquaring(cv::Mat& output_map, cv::Mat& SVF, int N);

void applyDeformationImage(cv::Mat& output_image, const cv::Mat& input_image, const cv::Mat& input_map);

void detect_salient_locations(std::set<cv::Point2f,comp_Point2f>& locations, const cv::Mat& image, const cv::Mat& mask);

void extract_unique_labels(std::set<unsigned char>& labels, const cv::Mat& label_image);

// Creates deformed_image and deformation_map such that  "deformed_image(deformation_map(x)) = input_image(x)"
void create_synthetic_transformation(cv::Mat& deformed_image, cv::Mat& deformation_map, const cv::Mat& input_image, const cv::Mat& input_mask, double std_gaussian, double max_svf, double reduction_factor, boost::random::mt19937& rng);

void create_mask_edge_object(cv::Mat& edge_map, const cv::Mat& label_image, unsigned char value_of_interest);

void load_true_transformation_from_deformation_map(std::map<cv::Point2f,cv::Point2f,comp_Point2f>& true_transformation, const cv::Mat& deformation_map, const cv::Mat& mask, unsigned char value_of_interest);

void load_locations_of_interest_from_mask(std::set<cv::Point2f,comp_Point2f>& locations, const cv::Mat& mask, unsigned char value_of_interest, int subsampling = 1);

void create_mask_object(cv::Mat& mask_object, const cv::Mat& label_image, unsigned char value_of_interest);

void load_landmark_file(std::vector<Eigen::Vector2d>& loaded_landmarks, const std::string& csv_filename);

void load_true_transformation_from_landmark_files(std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& true_transformation, const std::string& landmarks_fixed_image_filename, const std::string& landmarks_moving_image_filename);

void convert_from_cv_to_eigen(std::set<Eigen::Vector2d,comp_Point<2>>& set_eigen, const std::set<cv::Point2f,comp_Point2f>& set_cv);

void convert_from_cv_to_eigen(std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& map_eigen, const std::map<cv::Point2f,cv::Point2f,comp_Point2f>& map_cv);

void convert_from_cv_to_eigen(std::vector<std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>>& vector_map_eigen, const std::vector<std::map<cv::Point2f,cv::Point2f,comp_Point2f>>& vector_map_cv);

void run_landmark_placement(QueryingStrategy baseline, const cv::Mat& fixed_image, const cv::Mat& moving_image, const std::string& pair_identifier, const std::string& experiment_identifier, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& true_transformation_cv, const std::set<Eigen::Vector2d,comp_Point<2>>& salient_points_cv, const std::set<Eigen::Vector2d,comp_Point<2>>& target_points_cv, const std::set<Eigen::Vector2d,comp_Point<2>>& evaluation_points_cv, const std::vector<Eigen::Vector2d>& queries_for_burn_in, boost::random::mt19937& rng,  PartialWindow& partial_window_fixed_image, PartialWindow& partial_window_moving_image, LandmarkAnnotator<2>& landmark_annotator, const std::string& results_folder, int nb_elastix_parameters, bool load_precomputed_registrations = false, int annotation_budget = 101);

void run_landmark_placement(QueryingStrategy baseline, const cv::Mat& fixed_image, const cv::Mat& moving_image, const std::string& pair_identifier, const std::string& experiment_identifier, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& true_transformation_cv, const std::vector<std::vector<Gaussian_Process_Observation<2>>>& prior_observations, const std::set<Eigen::Vector2d,comp_Point<2>>& salient_points_cv, const std::set<Eigen::Vector2d,comp_Point<2>>& target_points_cv, const std::set<Eigen::Vector2d,comp_Point<2>>& evaluation_points_cv, const std::vector<Eigen::Vector2d>& queries_for_burn_in, boost::random::mt19937& rng,  PartialWindow& partial_window_fixed_image, PartialWindow& partial_window_moving_image, LandmarkAnnotator<2>& landmark_annotator, const std::string& results_folder, int nb_elastix_parameters, bool load_precomputed_registrations = false, int annotation_budget = 101);


//void read_transformix_output_points(std::map<cv::Point2f,cv::Point2f,comp_Point2f>& transformation, const std::string& output_points_filename);

//void write_transformix_input_points(const std::set<cv::Point2f,comp_Point2f>& target_points, const std::string& input_points_filename);

//void register_and_transform_with_elastix(std::map<cv::Point2f,cv::Point2f,comp_Point2f>& transformation, std::set<cv::Point2f,comp_Point2f>& target_points, const std::string& ///fixed_image_filename, const std::string& moving_image_filename, const std::string& temp_folder, const std::string& elastix_registration_parameters, bool load_precomputed_file);

void register_and_transform_with_elastix(std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, std::set<Eigen::Vector2d,comp_Point<2>>& target_points, const cv::Mat& fixed_image, const cv::Mat& moving_image, const std::string& pair_identifier, const std::string& temp_folder, const std::string& elastix_registration_parameters, bool load_precomputed_file,  bool remove_created_files, CoordinateType coordinate_type);

//void add_random_noise_transformation(std::map<cv::Point2f,cv::Point2f,comp_Point2f>& output_transformation, const std::map<cv::Point2f,cv::Point2f,comp_Point2f>& true_transformation, double standard_deviation, boost::random::mt19937& rng);

void tranform_color_image_with_elastix(const std::string& output_image_filename, const std::string& input_image_filename, const std::string& elastix_transformation_file, const std::string& temp_folder);

void read_transformix_output_points(cv::Mat& transformation, const std::string& output_points_filename);

void run_landmarks_histology();

void run_on_nissl_oct();

void example_cima_dataset();

#endif // REGISTRATION_AVERAGING_INCLUDED
