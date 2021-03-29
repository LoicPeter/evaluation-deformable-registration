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


#include "human_landmark_annotator.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/gamma.hpp>

HumanLandmarkAnnotator_2D::HumanLandmarkAnnotator_2D(const std::string& database_filename, cv::Mat *fixed_image, cv::Mat *moving_image, PartialWindow *partial_window_fixed_image, PartialWindow *partial_window_moving_image, double default_radius, double confidence_level)
{
    m_fixed_image_ptr = fixed_image;
    m_moving_image_ptr = moving_image;
    m_partial_window_fixed_image_ptr = partial_window_fixed_image;
    m_partial_window_moving_image_ptr = partial_window_moving_image;
    m_default_radius = default_radius;
    m_confidence_level = confidence_level;
    m_annotation_database.load(database_filename);
    m_database_filename = database_filename;
}

HumanLandmarkAnnotator_2D::~HumanLandmarkAnnotator_2D()
{
    m_annotation_database.save(m_database_filename);
}

void HumanLandmarkAnnotator_2D::getAnnotation(Gaussian_Process_Observation<2>& obs, bool& is_observation_uncertain, bool& no_point_provided, const Eigen::Vector2d& queried_location, const Eigen::Vector2d& predicted_location)
{
    obs.input_point = queried_location;
    if (m_annotation_database.is_obs_available(obs.input_point))
    {
        std::cout << "Observation available" << std::endl;
        m_annotation_database.retrieve_observation(obs.output_point,obs.observation_noise_covariance_matrix,obs.input_point);
        std::cout << "Observation retrieved from database" << std::endl;
    }
    else
    {
        std::cout << "Observation not available" << std::endl;
        query_2D_annotation(obs.output_point,obs.observation_noise_covariance_matrix,no_point_provided,obs.input_point,*m_fixed_image_ptr,*m_moving_image_ptr,*m_partial_window_fixed_image_ptr,*m_partial_window_moving_image_ptr,predicted_location,m_default_radius,m_confidence_level);
        if (!no_point_provided)
            m_annotation_database.add_to_database(obs);
    }
    
    std::cout << "So far, all locations that are manually annotated are considered as uncertain" << std::endl;
    is_observation_uncertain = true;
}

void query_2D_ellipse(cv::Point2f& centre, cv::Point2f& pt_a, cv::Point2f& pt_b, bool& no_point_provided, const cv::Mat& input_frame, const std::string& window_name, double default_radius)
{
    // Parameters
    int thickness_line = 2;
    int thickness_ellipse = 2;
    int line_type = cv::LINE_8;
    cv::Scalar color_centre(0,0,255);
    cv::Scalar color_line(255,0,0);
    cv::Scalar color_ellipse(0,255,0);
    int radius_centre = 2;
    int thickness_centre = 2;

    // Show image
    cv::namedWindow(window_name);
    cv::imshow(window_name,input_frame);
    cv::Mat current_displayed_frame; 
    
    // Collect the landmarks from user interaction
    bool exit_requested(false);
    int entered_key;
    no_point_provided = false;

    // Mouse parameters
    MouseParamsEllipseAnnotation mouse_params;
    bool draw_centre = false;
    bool draw_first_axis = false;
    mouse_params.drawing_phase = place_centre;
    cv::setMouseCallback(window_name, onMouse_2DEllipse, (void*)(&mouse_params));

    // First phase: place centre of ellipse
    while ((exit_requested==false) && (mouse_params.drawing_phase!=over))
    {
        // Show image
        input_frame.copyTo(current_displayed_frame);
        
        // Current point 
        cv::Point2f current_pt(mouse_params.current_x,mouse_params.current_y);
        
        // If the centre was placed, show it
        if (draw_centre)
            cv::circle(current_displayed_frame,centre,radius_centre,color_centre,thickness_centre);
        
        // If the firs axis was placed, show it
        if (draw_first_axis)
             cv::line(current_displayed_frame,2*centre - pt_a,pt_a,color_line,thickness_line,line_type);
        
        if (mouse_params.drawing_phase == place_centre)
            centre = current_pt;
        
        if (mouse_params.drawing_phase == place_first_axis)
        {
            draw_centre = true;
            pt_a = current_pt;
            if (pt_a == centre)
                pt_a = centre + cv::Point2f(default_radius,0); // avoid degenerate ellipse
            //cv::Point2f current_pt(mouse_params.current_x,mouse_params.current_y);
            cv::line(current_displayed_frame,2*centre - current_pt,current_pt,color_line,thickness_line,line_type);
        }
        
        if (mouse_params.drawing_phase == place_second_axis)
        {
            draw_centre = true;
            draw_first_axis = true;
            cv::Point2f v_1, v_2;
            double current_v_2_coordinate;
            
            v_1 = pt_a - centre;
            v_1 = (1/(cv::norm(v_1)))*v_1;
            std::cout << "v_1 = " << v_1 << std::endl; 
            v_2 = cv::Point2f(v_1.y,-(v_1.x));
            current_v_2_coordinate = v_2.dot(current_pt - centre);
            pt_b = centre + current_v_2_coordinate*v_2;
            
            // Avoid degenerate case (line)
            if (pt_b == centre)
                pt_b = centre + default_radius*v_2;
            
            double theta_rad = atan2(v_1.y,v_1.x);
            double theta_deg = theta_rad*180/CV_PI;
            if (theta_deg<0)
                theta_deg = 360 + theta_deg;
            
            cv::ellipse(current_displayed_frame,centre, cv::Size(cv::norm(pt_a - centre),abs(current_v_2_coordinate)),theta_deg,0,360,color_ellipse,thickness_ellipse,line_type); 
            cv::line(current_displayed_frame,2*centre - pt_b,pt_b,color_line,thickness_line,line_type);
        }
        
        cv::imshow(window_name,current_displayed_frame);
        entered_key = (int)cv::waitKey(1);

        if ((entered_key==27) || (mouse_params.drawing_phase == over)) // escape
            exit_requested = true;
    }

    // Process the reason why the loop was left
    if (mouse_params.drawing_phase == place_centre)
    {
        std::cout << "No centre provided - the user left the interface" << std::endl;
        no_point_provided = true;
        centre = cv::Point2f(-1,-1);
    }
    else
    {
        if (mouse_params.drawing_phase != over)
        {
            std::cout << "No ellipse provided - we use by default an uncertainty disk of radius " << default_radius << " pixels" << std::endl;
            pt_a = centre + cv::Point2f(default_radius,0);
            pt_b = centre + cv::Point2f(0,default_radius);
        }
    }
    

    
    //Sanity check based on visualisation / sampling 
    Eigen::Vector2d mean;
    Eigen::Matrix2d cov_matrix;
    convert_2D_ellipse_to_2D_gaussian(mean,cov_matrix,centre,pt_a,pt_b,0.01);
    
    // Sample for sanity check
    int nb_samples = 1000;
    boost::random::mt19937 rng;
    Eigen::MatrixXd L = cov_matrix.llt().matrixL();;
    boost::normal_distribution<double> normal_dist(0.0,1.0);
    Eigen::VectorXd current_sample(2), normal_sample(2);
    for (int ind=0; ind<nb_samples; ++ind)
    {
        // Sample normal random variables
        for (int d=0; d<2; ++d)
            normal_sample(d) = normal_dist(rng);

        // This is the sample for the desired multivariate distrbution
        current_sample = mean + L*normal_sample;
        
        cv::circle(current_displayed_frame,cv::Point2f(current_sample(0),current_sample(1)),2,color_centre,1);
        cv::imshow(window_name,current_displayed_frame);
        cv::waitKey(1);
    }
    cv::waitKey(-1);
}

void query_2D_ellipse(cv::Point2f& centre, cv::Point2f& pt_a, cv::Point2f& pt_b, bool& no_point_provided, const cv::Mat& input_frame, const PartialWindow& partial_window, double default_radius)
{
    // Parameters
    int thickness_line = 2;
    int thickness_ellipse = 2;
    int line_type = cv::LINE_8;
    cv::Scalar color_centre(0,0,255);
    cv::Scalar color_line(255,0,0);
    cv::Scalar color_ellipse(0,255,0);
    int radius_centre = 2;
    int thickness_centre = 2;

    // Show image
    imshow(partial_window,input_frame);
    cv::Mat current_displayed_frame; 
    
    // Collect the landmarks from user interaction
    bool exit_requested(false);
    int entered_key;
    no_point_provided = false;

    // Mouse parameters
    MouseParamsEllipseAnnotation mouse_params;
    bool draw_centre = false;
    bool draw_first_axis = false;
    mouse_params.drawing_phase = place_centre;
    cv::setMouseCallback(partial_window.name, onMouse_2DEllipse, (void*)(&mouse_params));

    // First phase: place centre of ellipse
    while ((exit_requested==false) && (mouse_params.drawing_phase!=over))
    {
        // Show image
        input_frame.copyTo(current_displayed_frame);
        
        // Current point 
        cv::Point2f current_pt(mouse_params.current_x + partial_window.current_origin[0],mouse_params.current_y + partial_window.current_origin[1]);
        
        // If the centre was placed, show it
        if (draw_centre)
            cv::circle(current_displayed_frame,centre,radius_centre,color_centre,thickness_centre);
        
        // If the firs axis was placed, show it
        if (draw_first_axis)
             cv::line(current_displayed_frame,2*centre - pt_a,pt_a,color_line,thickness_line,line_type);
        
        if (mouse_params.drawing_phase == place_centre)
            centre = current_pt;
        
        if (mouse_params.drawing_phase == place_first_axis)
        {
            draw_centre = true;
            pt_a = current_pt;
            cv::line(current_displayed_frame,2*centre - current_pt,current_pt,color_line,thickness_line,line_type);
        }
        
        if (mouse_params.drawing_phase == place_second_axis)
        {
            draw_centre = true;
            draw_first_axis = true;
            cv::Point2f v_1, v_2;
            double current_v_2_coordinate;
            v_1 = pt_a - centre;
            v_1 = (1/(cv::norm(v_1)))*v_1;
            v_2 = cv::Point2f(v_1.y,-(v_1.x));
            current_v_2_coordinate = v_2.dot(current_pt - centre);
            pt_b = centre + current_v_2_coordinate*v_2;
            
            double theta_rad = atan2(v_1.y,v_1.x);
            double theta_deg = theta_rad*180/CV_PI;
            if (theta_deg<0)
                theta_deg = 360 + theta_deg;
            
            cv::ellipse(current_displayed_frame,centre, cv::Size(cv::norm(pt_a - centre),abs(current_v_2_coordinate)),theta_deg,0,360,color_ellipse,thickness_ellipse,line_type); 
            cv::line(current_displayed_frame,2*centre - pt_b,pt_b,color_line,thickness_line,line_type);
        }

        imshow(partial_window,current_displayed_frame);
        entered_key = (int)cv::waitKey(1);

        if ((entered_key==27) || (mouse_params.drawing_phase == over)) // escape
            exit_requested = true;
    }

    // Process the reason why the loop was left
    if (mouse_params.drawing_phase == place_centre)
    {
        std::cout << "No centre provided - the user left the interface" << std::endl;
        no_point_provided = true;
        centre = cv::Point2f(-1,-1);
    }
    else
    {
        if (mouse_params.drawing_phase != over)
        {
            std::cout << "No ellipse provided - we use by default an uncertainty disk of radius " << default_radius << " pixels" << std::endl;
            pt_a = centre + cv::Point2f(default_radius,0);
            pt_b = centre + cv::Point2f(0,default_radius);
        }
    }
    
//     //Sanity check based on visualisation / sampling 
//     Eigen::VectorXd mean;
//     Eigen::MatrixXd cov_matrix;
//     convert_2D_ellipse_to_2D_gaussian(mean,cov_matrix,centre,pt_a,pt_b,0.01);
//     
//     // Sample for sanity check
//     int nb_samples = 1000;
//     boost::random::mt19937 rng;
//     Eigen::MatrixXd L = cov_matrix.llt().matrixL();;
//     boost::normal_distribution<double> normal_dist(0.0,1.0);
//     Eigen::VectorXd current_sample(2), normal_sample(2);
//     for (int ind=0; ind<nb_samples; ++ind)
//     {
//         // Sample normal random variables
//         for (int d=0; d<2; ++d)
//             normal_sample(d) = normal_dist(rng);
// 
//         // This is the sample for the desired multivariate distrbution
//         current_sample = mean + L*normal_sample;
//         
//         cv::circle(current_displayed_frame,cv::Point2f(current_sample(0),current_sample(1)),2,color_centre,1);
//         imshow(partial_window,current_displayed_frame);
//         cv::waitKey(1);
//     }
//     cv::waitKey(-1);
}

void onMouse_2DEllipse(int evt, int x, int y, int flags, void* mouse_params)
{
    MouseParamsEllipseAnnotation* ptPtr = (MouseParamsEllipseAnnotation*)mouse_params;
    
    ptPtr->current_x = x;
    ptPtr->current_y = y;
    
    if (ptPtr->drawing_phase == place_centre)
    {
        if (evt == CV_EVENT_LBUTTONDOWN)
        {
            ptPtr->drawing_phase = place_first_axis;
        }
    }
    else
    {
        if (ptPtr->drawing_phase == place_first_axis)
        {

            if (evt == CV_EVENT_LBUTTONDOWN)
            {
                ptPtr->drawing_phase = place_second_axis;
            }
        }
        else
        {
            if (ptPtr->drawing_phase == place_second_axis)
            {
                
                if (evt == CV_EVENT_LBUTTONDOWN)
                {
                    ptPtr->drawing_phase = over;
                }
            }
        }
    }
}


void query_2D_annotation(Eigen::Vector2d& output_point, Eigen::Matrix2d& observation_noise_covariance_matrix, bool& no_point_provided, const Eigen::Vector2d& input_point, const cv::Mat& fixed_image, const cv::Mat& moving_image, const PartialWindow& partial_window_fixed_image, const PartialWindow& partial_window_moving_image, const Eigen::Vector2d& predicted_location, double default_radius, double confidence_level)
{
    cv::Point2f input_point_cv(input_point(0),input_point(1)), predicted_location_cv(predicted_location(0),predicted_location(1));
    
    // Show fixed_image with queried location
    cv::Mat fixed_image_with_landmark;
    fixed_image.copyTo(fixed_image_with_landmark);
    cv::circle(fixed_image_with_landmark,input_point_cv,2,cv::Scalar(0,150,0),2);
    partial_window_fixed_image.centre_on_point(input_point_cv,fixed_image_with_landmark);
    imshow(partial_window_fixed_image,fixed_image_with_landmark);
    
    // Centre moving image on predicted location
    partial_window_moving_image.centre_on_point(predicted_location_cv,moving_image);
    
    // Query ellipse
    cv::Point2f ellipse_centre, ellipse_pt_a, ellipse_pt_b;
    query_2D_ellipse(ellipse_centre,ellipse_pt_a,ellipse_pt_b,no_point_provided,moving_image,partial_window_moving_image,default_radius);

    if (no_point_provided==false)
        convert_2D_ellipse_to_2D_gaussian(output_point,observation_noise_covariance_matrix,ellipse_centre,ellipse_pt_a,ellipse_pt_b,confidence_level);
}


void convert_2D_ellipse_to_2D_gaussian(Eigen::Vector2d& mean, Eigen::Matrix2d& cov_matrix, const cv::Point2f& centre, const cv::Point2f& pt_a, const cv::Point2f& pt_b, double confidence_level)
{
    double d = 2; // degrees of reedom
    double gamma_square = 2*boost::math::gamma_p_inv(d/2,1 - confidence_level);
    
    mean(0) = centre.x;
    mean(1) = centre.y; 
    
    
    Eigen::Matrix2d V, D;
    cv::Point2f v_1 = pt_a - centre;
    double a_1 = cv::norm(pt_a - centre);
    v_1 = (1/a_1)*v_1;
    cv::Point2f v_2 = cv::Point2f(v_1.y,-(v_1.x));
    double a_2 = cv::norm(pt_b - centre);
    
    // Fill V
    V(0,0) = v_1.x;
    V(1,0) = v_1.y;
    V(0,1) = v_2.x;
    V(1,1) = v_2.y;
    
    // Fill D
    D(0,1) = 0;
    D(1,0) = 0;
    D(0,0) = a_1*a_1/gamma_square;
    D(1,1) = a_2*a_2/gamma_square;
    
    cov_matrix = V*D*(V.transpose());
}


