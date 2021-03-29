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


#ifndef HUMAN_LANDMARK_ANNOTATOR_H_INCLUDED
#define HUMAN_LANDMARK_ANNOTATOR_H_INCLUDED

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "landmark_annotator.hpp"
#include "gp_observation_database.hpp"
#include "partial_window.hpp"
#include <Eigen/Dense>


class HumanLandmarkAnnotator_2D : public LandmarkAnnotator<2>
{
    public:
    HumanLandmarkAnnotator_2D(const std::string& database_filename, cv::Mat *fixed_image, cv::Mat *moving_image, PartialWindow *partial_window_fixed_image, PartialWindow *partial_window_moving_image, double default_radius, double confidence_level);
    ~HumanLandmarkAnnotator_2D();
    void getAnnotation(Gaussian_Process_Observation<2>& obs, bool& is_observation_uncertain, bool& no_point_provided, const Eigen::Vector2d& queried_location, const Eigen::Vector2d& predicted_location);
    LandmarkAnnotator* clone() const; 
    
    
    private:
    GP_Observation_Database<2> m_annotation_database;
    cv::Mat *m_fixed_image_ptr;
    cv::Mat *m_moving_image_ptr;
    PartialWindow *m_partial_window_fixed_image_ptr;
    PartialWindow *m_partial_window_moving_image_ptr;
    double m_default_radius;
    double m_confidence_level;
    std::string m_database_filename;
    
};

enum EllipseDrawingPhase_2D {place_centre, place_first_axis, place_second_axis, over};

struct MouseParamsEllipseAnnotation
{
  //  std::vector<cv::Point2f> landmarks;
    EllipseDrawingPhase_2D drawing_phase;
    int current_x;
    int current_y;
};

void query_2D_ellipse(cv::Point2f& centre, cv::Point2f& pt_a, cv::Point2f& pt_b, bool& no_point_provided, const cv::Mat& input_frame, const std::string& window_name, double default_radius = 0.5);


void onMouse_2DEllipse(int evt, int x, int y, int flags, void* mouse_params);


void query_2D_ellipse(cv::Point2f& centre, cv::Point2f& pt_a, cv::Point2f& pt_b, bool& no_point_provided, const cv::Mat& input_frame, const PartialWindow& partial_window, double default_radius = 0.5);

void query_2D_annotation(Eigen::Vector2d& output_point, Eigen::Matrix2d& observation_noise_covariance_matrix, bool& no_point_provided, const Eigen::Vector2d& input_point, const cv::Mat& fixed_image, const cv::Mat& moving_image, const PartialWindow& partial_window_fixed_image, const PartialWindow& partial_window_moving_image, const Eigen::Vector2d& predicted_location, double default_radius, double confidence_level);

void convert_2D_ellipse_to_2D_gaussian(Eigen::Vector2d& mean, Eigen::Matrix2d& cov_matrix, const cv::Point2f& centre, const cv::Point2f& pt_a, const cv::Point2f& pt_b, double confidence_level = 0.01);







#endif
