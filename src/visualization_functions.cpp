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


#include "visualization_functions.hpp"

#include <iostream>

cv::Point2f eigen_to_cv(const Eigen::Vector2d& eigen_point)
{
    return cv::Point2f(eigen_point(0),eigen_point(1));
}


void plot_landmarks_on_image(cv::Mat& image, const std::set<Eigen::Vector2d,comp_Point<2>>& landmarks, int radius, const cv::Scalar& color, int thickness)
{
    std::set<Eigen::Vector2d,comp_Point<2>>::const_iterator it;
    for (it=landmarks.begin(); it!=landmarks.end(); ++it)
    {
        cv::circle(image,eigen_to_cv(*it),radius,color,thickness);
    }
}

void plot_landmarks_on_image(cv::Mat& image, const std::set<Eigen::Vector2d,comp_Point<2>>& landmarks, int markerType, int markerSize, const cv::Scalar& color, int thickness)
{
    std::set<Eigen::Vector2d,comp_Point<2>>::const_iterator it;
    for (it=landmarks.begin(); it!=landmarks.end(); ++it)
        cv::drawMarker(image,eigen_to_cv(*it),color,markerType,markerSize,thickness);
}

void plot_salient_and_target_points(const cv::Mat& input_image, const std::set<Eigen::Vector2d,comp_Point<2>>& salient_points, const std::set<Eigen::Vector2d,comp_Point<2>>& target_points, const std::string& filename)
{
    cv::Mat output_image;
    cv::normalize(input_image,output_image,0,255,cv::NORM_MINMAX);
    if (input_image.channels()==1) // convert to color image if the input image is grayscale
    {
        std::vector<cv::Mat> three_output_image(3);
        for (int k=0; k<3; ++k)
            three_output_image[k] = output_image;
        cv::merge(three_output_image,output_image);
    }
    plot_landmarks_on_image(output_image,target_points,0.5,cv::Scalar(205,0,0),1);
    plot_landmarks_on_image(output_image,salient_points,cv::MARKER_CROSS,6,cv::Scalar(0,128,255),2);
    cv::imwrite(filename,output_image);
}

void normalise_and_write(const cv::Mat& input_image, const std::string& filename)
{
    cv::Mat output_image;
    cv::normalize(input_image,output_image,0,255,cv::NORM_MINMAX);
    cv::imwrite(filename,output_image);
}


void plot_2D_transformation(const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, const std::string& window_name, const cv::Scalar& color, int nb_rows, int nb_cols)
{
    cv::Mat background_image(nb_rows,nb_cols,CV_8UC1,cv::Scalar::all(0)), output_image;
    plot_2D_transformation(output_image,background_image,transformation,window_name,color);
}

// Plots a field
void plot_2D_transformation(cv::Mat& output_image, const cv::Mat& background_image, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, const std::string& window_name, const cv::Scalar& color)
{
    std::set<Eigen::Vector2d,comp_Point<2>> locations_in_map = create_set_from_map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>(transformation);
    plot_2D_transformation(output_image,background_image,transformation,locations_in_map,window_name,color);
}

void plot_2D_transformation(cv::Mat& output_image, const cv::Mat& background_image, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, const std::set<Eigen::Vector2d,comp_Point<2>>& locations, const std::string& window_name, const cv::Scalar& color)
{
    background_image.copyTo(output_image);
    std::set<Eigen::Vector2d,comp_Point<2>>::const_iterator it;
    int thickness = 3;
    int line_type = 8;
    int shift = 0;
    double tip_length = 0.1;
    for (it=locations.begin(); it!=locations.end(); ++it)
    {
 //       std::cout << it->first << " " << it->second << std::endl;
        cv::arrowedLine(output_image,eigen_to_cv(*it),eigen_to_cv(transformation.at(*it)),color,thickness,line_type,shift,tip_length);
    }
    cv::imshow(window_name,output_image);
   // cv::waitKey(500);
}



void create_heat_map_outliers(cv::Mat& heat_map, Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, const std::set<Eigen::Vector2d,comp_Point<2>>& target_set, int nb_rows, int nb_cols)
{
    heat_map = cv::Mat(nb_rows,nb_cols,CV_64F,cv::Scalar::all(0));
    Eigen::Vector2d mean_location, predicted_location;
    Eigen::Matrix2d covariance_matrix;
    for (std::set<Eigen::Vector2d,comp_Point<2>>::const_iterator it=target_set.begin(); it!=target_set.end(); ++it)
    {
        gp.mean(mean_location,*it);
        gp.covariance(covariance_matrix,*it,*it);
        predicted_location = transformation_to_evaluate.at(*it);
        double score = compute_outlier_score_for_evaluation<2>(predicted_location,mean_location,covariance_matrix);
        double x = (*it)(0);
        double y = (*it)(1);
        heat_map.at<double>(y,x) = score;
    }
}

void plot_heat_map_outliers(Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, const std::set<Eigen::Vector2d,comp_Point<2>>& target_set, int nb_rows, int nb_cols, const std::string& window_name)
{
    cv::Mat heat_map, heat_map_uchar, heat_map_jet;
    create_heat_map_outliers(heat_map,gp,transformation_to_evaluate,target_set,nb_rows,nb_cols);
    heat_map = 255*heat_map;
    heat_map.convertTo(heat_map_uchar,CV_8U);
   
    cv::applyColorMap(heat_map_uchar,heat_map_jet,cv::COLORMAP_JET);
    cv::imshow(window_name,heat_map_jet);
    cv::waitKey(1);
}


void create_heat_map_outliers_at_predefined_candidates(cv::Mat& heat_map, Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, int nb_rows, int nb_cols)
{
    heat_map = cv::Mat(nb_rows,nb_cols,CV_64F,cv::Scalar::all(0));
    Eigen::Vector2d predicted_location, mean_location;
    
    std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>> mean_locations;
    std::map<Eigen::Vector2d,Eigen::Matrix2d,comp_Point<2>> covariance_matrices;
    
    std::cout << "Compute dense mean transformation" << std::endl;
    gp.predict_transformation_at_predefined_candidates(mean_locations,ind_predefined_target_set);
    
    std::cout << "Compute dense covariance matrices" << std::endl;
    gp.compute_covariance_matrix_at_predefined_candidates(covariance_matrices,ind_predefined_target_set);
    std::cout << "Done" << std::endl;
        
    for (std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>::const_iterator it=mean_locations.begin(); it!=mean_locations.end(); ++it)
    {
        predicted_location = transformation_to_evaluate.at(it->first);
        mean_location = it->second;
        double score = compute_outlier_score_for_evaluation<2>(predicted_location,mean_location,covariance_matrices.at(it->first));
        double x = (it->first)(0);
        double y = (it->first)(1);
        heat_map.at<double>(y,x) = score;
 //       std::cout << it->first.x << " " << it->first.y << " " << score << std::endl;
    }
}

void create_heat_map_entropy_at_predefined_candidates(cv::Mat& heat_map, Gaussian_Process<2>& gp, int ind_predefined_target_set, int nb_rows, int nb_cols)
{
    heat_map = cv::Mat(nb_rows,nb_cols,CV_64F,cv::Scalar::all(0));
    std::map<Eigen::Vector2d,Eigen::Matrix2d,comp_Point<2>> covariance_matrices;
    gp.compute_covariance_matrix_at_predefined_candidates(covariance_matrices,ind_predefined_target_set);
        
    for (std::map<Eigen::Vector2d,Eigen::Matrix2d,comp_Point<2>>::const_iterator it=covariance_matrices.begin(); it!=covariance_matrices.end(); ++it)
    {
        double score = 1 + std::log(2*M_PI) + 0.5*std::log(covariance_matrices.at(it->first).determinant());
        double x = (it->first)(0);
        double y = (it->first)(1);
        heat_map.at<double>(y,x) = score;
    }
}


void plot_heat_map_outliers_at_predefined_candidates(Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, int nb_rows, int nb_cols, const std::string& window_name, int subsampling_display, const std::string& heat_map_folder, const std::string pair_identifier)
{
    cv::Mat heat_map, heat_map_uchar, heat_map_resized, heat_map_jet;
    create_heat_map_outliers_at_predefined_candidates(heat_map,gp,transformation_to_evaluate,ind_predefined_target_set,nb_rows,nb_cols);
    heat_map.convertTo(heat_map_uchar,CV_8UC1,255.0);
  //  std::cout << heat_map_uchar << std::endl;
    cv::resize(heat_map_uchar,heat_map_resized,cv::Size(),1/((double)subsampling_display),1/((double)subsampling_display),cv::INTER_NEAREST);
 //   std::cout << heat_map_resized << std::endl;
    cv::applyColorMap(heat_map_resized,heat_map_jet,cv::COLORMAP_JET);
  //  std::cout << heat_map_jet << std::endl;
    std::cout << heat_map_folder + "/" + pair_identifier + ".png";
    cv::imwrite(heat_map_folder + "/" + pair_identifier + ".png",heat_map_jet);
    cv::imshow(window_name,heat_map_jet);
    cv::waitKey(1);
}

void plot_heat_map_entropy_at_predefined_candidates(Gaussian_Process<2>& gp, int ind_predefined_target_set, int nb_rows, int nb_cols, const std::string& window_name, int subsampling_display, const std::string& heat_map_folder, const std::string pair_identifier)
{
    cv::Mat heat_map, heat_map_normalised, heat_map_uchar, heat_map_resized, heat_map_jet;
    create_heat_map_entropy_at_predefined_candidates(heat_map,gp,ind_predefined_target_set,nb_rows,nb_cols);
    cv::resize(heat_map,heat_map_resized,cv::Size(),1/((double)subsampling_display),1/((double)subsampling_display),cv::INTER_NEAREST);
    //std::cout << heat_map_resized << std::endl;
    cv::normalize(heat_map_resized,heat_map_normalised,0,1,cv::NORM_MINMAX);
    heat_map_normalised.convertTo(heat_map_uchar,CV_8UC1,255.0);

    cv::applyColorMap(heat_map_uchar,heat_map_jet,cv::COLORMAP_JET);
  //  std::cout << heat_map_jet << std::endl;
    std::cout << heat_map_folder + "/" + pair_identifier + ".png";
    cv::imwrite(heat_map_folder + "/" + pair_identifier + ".png",heat_map_jet);
  //  cv::imshow(window_name,heat_map_jet);
  //  cv::waitKey(1);
}

void plot_joint_heat_map_at_predefined_candidates(Gaussian_Process<2>& gp, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, int nb_rows, int nb_cols, const cv::Mat fixed_image, const std::string& window_name, int subsampling_display, const std::string& joint_map_folder, const std::string pair_identifier)
{
    // Error map
    cv::Mat error_map, error_map_uchar, error_map_resized, error_map_jet;
    create_heat_map_outliers_at_predefined_candidates(error_map,gp,transformation_to_evaluate,ind_predefined_target_set,nb_rows,nb_cols);
    error_map.convertTo(error_map_uchar,CV_8UC1,255.0);
    cv::resize(error_map_uchar,error_map_resized,cv::Size(),1/((double)subsampling_display),1/((double)subsampling_display),cv::INTER_NEAREST);
    cv::applyColorMap(error_map_resized,error_map_jet,cv::COLORMAP_JET);
    
    // Entropy map
    cv::Mat entropy_map, entropy_map_normalised, entropy_map_uchar, entropy_map_resized, entropy_map_jet;
    create_heat_map_entropy_at_predefined_candidates(entropy_map,gp,ind_predefined_target_set,nb_rows,nb_cols);
    cv::resize(entropy_map,entropy_map_resized,cv::Size(),1/((double)subsampling_display),1/((double)subsampling_display),cv::INTER_NEAREST);
    cv::normalize(entropy_map_resized,entropy_map_normalised,0,1,cv::NORM_MINMAX);
    entropy_map_normalised.convertTo(entropy_map_uchar,CV_8UC1,255.0);
    cv::applyColorMap(entropy_map_uchar,entropy_map_jet,cv::COLORMAP_JET);
    
    // For transparency
    unsigned char alpha_blending = 0.5;
    cv::Mat a_channel(entropy_map_uchar.rows,entropy_map_uchar.cols,CV_8UC1,cv::Scalar::all(alpha_blending));
    cv::Mat zero_map(entropy_map_uchar.rows,entropy_map_uchar.cols,CV_8UC1,cv::Scalar::all(0));
    cv::Mat ones_map_3d(entropy_map_uchar.rows,entropy_map_uchar.cols,CV_64FC3,cv::Scalar::all(1));
    
    // Fixed image
    cv::Mat fixed_image_resized, fixed_image_gray, fixed_image_3d, fixed_image_float;
    cv::cvtColor(fixed_image,fixed_image_gray,CV_RGB2GRAY);
    cv::resize(fixed_image_gray,fixed_image_resized,cv::Size(),1/((double)subsampling_display),1/((double)subsampling_display),cv::INTER_NEAREST);
    std::vector<cv::Mat> fixed_image_3d_channels(3);
    fixed_image_3d_channels[0] = fixed_image_resized;
    fixed_image_3d_channels[1] = fixed_image_resized;
    fixed_image_3d_channels[2] = fixed_image_resized;
    cv::merge(fixed_image_3d_channels,fixed_image_3d);
    fixed_image_3d.convertTo(fixed_image_float,CV_64FC3,1.0/255.0);

    std::vector<cv::Mat> entropy_normalised_3d_channels(3);
    entropy_normalised_3d_channels[0] = entropy_map_normalised;
    entropy_normalised_3d_channels[1] = entropy_map_normalised;
    entropy_normalised_3d_channels[2] = entropy_map_normalised;
    cv::Mat entropy_normalised_3d;
    cv::merge(entropy_normalised_3d_channels,entropy_normalised_3d);
    
    // Weight by the entropy
    cv::Mat fixed_image_float_times_entropy = fixed_image_float.mul(entropy_normalised_3d);
    cv::Mat error_map_jet_float, error_map_jet_float_times_entropy;
    error_map_jet.convertTo(error_map_jet_float,CV_64FC3,1.0/255.0);
    error_map_jet_float_times_entropy = error_map_jet_float.mul(ones_map_3d - entropy_normalised_3d);
    
    // Final blend
    cv::Mat final_blend;
    cv::Mat final_blend_float = 0.3*fixed_image_float + 0.7*(error_map_jet_float_times_entropy + fixed_image_float_times_entropy);
    final_blend_float.convertTo(final_blend,CV_8UC3,255.0);
    cv::imwrite(joint_map_folder + "/" + pair_identifier + "-JointMap.png",final_blend);
    cv::imwrite(joint_map_folder + "/" + pair_identifier + "-EntropyMap.png",entropy_map_jet);
    cv::imwrite(joint_map_folder + "/" + pair_identifier + "-ErrorMap.png",error_map_jet);
}

void plot_heat_map_entropy_at_predefined_candidates(Gaussian_Process<2>& gp_on_X, Gaussian_Process<2>& gp_on_T_and_X, int ind_predefined_target_set_on_X, int ind_predefined_target_set_T_on_X, int nb_rows, int nb_cols, const std::string& window_name, int subsampling_display, const std::string& heat_map_folder, const std::string pair_identifier)
{
    cv::Mat heat_map, heat_map_on_X, heat_map_on_T_and_X, heat_map_normalised, heat_map_uchar, heat_map_resized, heat_map_jet;
    create_heat_map_entropy_at_predefined_candidates(heat_map_on_X,gp_on_X,ind_predefined_target_set_on_X,nb_rows,nb_cols);
    create_heat_map_entropy_at_predefined_candidates(heat_map_on_T_and_X,gp_on_T_and_X,ind_predefined_target_set_T_on_X,nb_rows,nb_cols);
    heat_map = heat_map_on_X - heat_map_on_T_and_X;
    cv::resize(heat_map,heat_map_resized,cv::Size(),1/((double)subsampling_display),1/((double)subsampling_display),cv::INTER_NEAREST);
    //std::cout << heat_map_resized << std::endl;
    cv::normalize(heat_map_resized,heat_map_normalised,0,1,cv::NORM_MINMAX);
    heat_map_normalised.convertTo(heat_map_uchar,CV_8UC1,255.0);

    cv::applyColorMap(heat_map_uchar,heat_map_jet,cv::COLORMAP_JET);
  //  std::cout << heat_map_jet << std::endl;
    std::cout << heat_map_folder + "/" + pair_identifier + ".png";
    cv::imwrite(heat_map_folder + "/" + pair_identifier + ".png",heat_map_jet);
  //  cv::imshow(window_name,heat_map_jet);
  //  cv::waitKey(1);
}



VisualisationHelper<2>::VisualisationHelper(const cv::Mat& fixed_image, const cv::Mat& moving_image, int subsampling_dense_maps, bool create_dense_maps, const Eigen::Vector2d& spacing, bool display)
{
    fixed_image.copyTo(m_fixed_image);
    fixed_image.copyTo(m_fixed_image_for_display);
    moving_image.copyTo(m_moving_image);
    m_entropy_display = cv::Mat(fixed_image.rows,fixed_image.cols,CV_64F,cv::Scalar::all(0));
    m_create_dense_maps = create_dense_maps;
    m_subsampling_dense_maps = subsampling_dense_maps;
    m_spacing = spacing;
    m_display = display;
    
    if (create_dense_maps)
    {
        for (int y=0; y<fixed_image.rows; y+=subsampling_dense_maps)
        {
            for (int x=0; x<fixed_image.cols; x+=subsampling_dense_maps)
                    m_image_domain_for_display.insert(Eigen::Vector2d(x,y));
        }
    }
}


void VisualisationHelper<2>::plot_transformation(const std::string& window_name, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& predicted_transformation, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& true_transformation, const std::set<Eigen::Vector2d,comp_Point<2>>& queried_locations, const std::set<Eigen::Vector2d,comp_Point<2>>& evaluation_locations)
{
    if (m_display)
    {
        plot_2D_transformation(m_fixed_image_for_display,m_fixed_image,true_transformation,evaluation_locations,window_name,cv::Scalar(0,0,255));
        plot_landmarks_on_image(m_fixed_image_for_display,queried_locations,5,cv::Scalar(0,120,255),3);
        plot_2D_transformation(m_fixed_image_for_display,m_fixed_image_for_display,predicted_transformation,evaluation_locations,window_name,cv::Scalar(0,125,0));
        cv::waitKey(200);
    }
}


void VisualisationHelper<2>::plot_heat_map(const std::string& window_name, const std::string& heat_map_folder, const std::string pair_identifier, const std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation_to_evaluate, int ind_predefined_target_set, Gaussian_Process<2>& gp_on_X)
{
    plot_joint_heat_map_at_predefined_candidates(gp_on_X,transformation_to_evaluate,ind_predefined_target_set,m_fixed_image.rows,m_fixed_image.cols,m_fixed_image,"Joint heat map",m_subsampling_dense_maps,heat_map_folder,pair_identifier);
    cv::waitKey(1);
}


