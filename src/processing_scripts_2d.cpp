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


#include "processing_scripts_2d.hpp"
#include "processing_scripts.hpp"
#include "gaussian_kernel.hpp"
#include "human_landmark_annotator.hpp"

std::string removeExtension(const std::string& inputString)
{
	std::string res;
	int length = (int)inputString.size();
	res = inputString.substr(0,length-4);
	return res;
}


void performScalingAndSquaring(cv::Mat& output_map, cv::Mat& SVF, int N)
{
    cv::Mat empty_map;
    int dimx = SVF.cols;
    int dimy = SVF.rows;

    cv::Mat SVF_double, output_map_double(dimy,dimx,CV_64FC2);
    SVF.convertTo(SVF_double,CV_64F);

    // Create identity map
    cv::Mat identity_map(dimy,dimx,CV_64FC2);
    for (int y=0; y<dimy; ++y)
    {
        double *row_ptr = identity_map.ptr<double>(y);
        for (int x=0; x<dimx; ++x)
        {
            row_ptr[2*x] = x;
            row_ptr[2*x+1] = y;
        }
    }

    cv::Mat SVF_displacement(dimy,dimx,CV_64FC2), initial_map(dimy,dimx,CV_64FC2);
    cv::subtract(SVF_double,identity_map,SVF_displacement);
    initial_map = identity_map + (1/(std::pow<double>(2,N)))*SVF_displacement;

    //std::cout << std::fixed;

    for (int k=0; k<N; ++k)
    {
        square_exponential_vector_field<double>(output_map_double,initial_map);
        if (k!=(N-1))
            output_map_double.copyTo(initial_map);
    }

    output_map_double.convertTo(output_map,CV_32F);

}

void applyDeformationImage(cv::Mat& output_image, const cv::Mat& input_image, const cv::Mat& input_map)
{
    cv::Mat empty_map;
    cv::remap(input_image,output_image,input_map,empty_map,cv::INTER_LINEAR);
}


void detect_salient_locations(std::set<cv::Point2f,comp_Point2f>& locations, const cv::Mat& image, const cv::Mat& mask)
{
    CV_Assert(mask.type()==CV_8UC1);
    locations.clear();
    
    // Detect SIFT features
    cv::Ptr<cv::xfeatures2d::SURF> sift = cv::xfeatures2d::SURF::create(1);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(image,mask,keypoints,descriptors,false);
    
    for (std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin(); it!=keypoints.end(); ++it)
    {
        int rounded_x = std::round(it->pt.x);
        int rounded_y = std::round(it->pt.y);
        if (mask.at<unsigned char>(rounded_y,rounded_x)==255)
        {
            cv::Point2f rounded_point(rounded_x,rounded_y);
            locations.insert(rounded_point);
        }
        //std::cout << it->pt << " ";
    }
//   //  std::cout << "Number locations: " << locations.size() << std::endl;
    // Check the detected keypoints
 //   cv::Mat im_keypoints;
 //   drawKeypoints(image,keypoints,im_keypoints,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
 //   imshow("Keypoints",im_keypoints);
 //   cv::waitKey(-1);
    
}



void extract_unique_labels(std::set<unsigned char>& labels, const cv::Mat& label_image)
{
    CV_Assert(label_image.type()==CV_8UC1);
    labels.clear();
    
    int dimx = label_image.rows;
    int dimy = label_image.cols;
    for (int y=0; y<dimy; ++y)
    {
        for (int x=0; x<dimx; ++x)
        {
            unsigned char val = label_image.at<unsigned char>(y,x);
            if (labels.count(val)==0)
                labels.insert(val);
        }
    }
}

// Creates deformed_image and deformation_map such that  "deformed_image(deformation_map(x)) = input_image(x)"
void create_synthetic_transformation(cv::Mat& deformed_image, cv::Mat& deformation_map, const cv::Mat& input_image, const cv::Mat& input_mask, double std_gaussian, double max_svf, double reduction_factor, boost::random::mt19937& rng)
{
    CV_Assert(input_mask.depth()==CV_8U);

    // Distribution on SVF
    boost::uniform_real<float> svf_distribution(-max_svf,max_svf);
    
    // Build the structuring element for the erosion (to keep edges of the brain undeformed)
    cv::Mat struct_el = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size((int)ceil(std_gaussian),(int)ceil(std_gaussian)));
    cv::Mat eroded_mask, closed_mask, smooth_mask;
 //   cv::morphologyEx(input_mask,closed_mask,cv::MORPH_CLOSE,struct_el);
 //   cv::imshow("Closed mask",closed_mask);
 //   cv::waitKey(-1);
    cv::erode(input_mask,eroded_mask,struct_el);
    cv::GaussianBlur(eroded_mask,smooth_mask,cv::Size(),std_gaussian,std_gaussian);

    int nb_rows = eroded_mask.rows;
    int nb_cols = eroded_mask.cols;

    std::cout << nb_rows << " " << nb_cols << std::endl;
   // cv::imshow("Eroded mask",eroded_mask);
   // cv::waitKey(-1);

    
    int nb_rows_downsized = std::round(reduction_factor*nb_rows);
    int nb_cols_downsized = std::round(reduction_factor*nb_cols);
    cv::Mat downsized_svf(nb_rows_downsized,nb_cols_downsized,CV_32FC2,cv::Scalar::all(0));
    
    // Create a small image where each pixel is ampled independently
    for (int y=0; y<nb_rows_downsized; ++y)
    {
        float *row_ptr = downsized_svf.ptr<float>(y);
        for (int x=0; x<nb_cols_downsized; ++x)
        {
            row_ptr[2*x] = svf_distribution(rng);
            row_ptr[2*x+1] = svf_distribution(rng);    
        }
    }
    cv::Mat sampled_svf;
    cv::resize(downsized_svf,sampled_svf,cv::Size(nb_rows,nb_cols));
    
    cv::Mat smooth_mask_two_channels;
    std::vector<cv::Mat> vec_for_merge(2);
    for (int c=0; c<2; ++c)
    {
        smooth_mask.convertTo(vec_for_merge[c],CV_32F);
        vec_for_merge[c] = (1/((float)255))*vec_for_merge[c];
    }
    cv::merge(vec_for_merge,smooth_mask_two_channels);

    sampled_svf = sampled_svf.mul(smooth_mask_two_channels);
    
    // Create identity map
    std::cout << "Create identity" << std::endl;
    cv::Mat identity_map(nb_rows,nb_cols,CV_32FC2,cv::Scalar::all(0));
    for (int y=0; y<nb_rows; ++y)
    {
        float *row_ptr_id = identity_map.ptr<float>(y);
        for (int x=0; x<nb_cols; ++x)
        {
            row_ptr_id[2*x] = (float)x;
            row_ptr_id[2*x+1] = (float)y;
        }
    }
    
    // Scale and square
    int N=10;
    std::cout << "Scaling and squaring" << std::endl;
    cv::Mat deformation_map_inv;
    cv::Mat id_plus_svf = identity_map + sampled_svf;
    performScalingAndSquaring(deformation_map_inv,id_plus_svf,N);
    applyDeformationImage(deformed_image,input_image,deformation_map_inv);
    
    // Deformation map
     std::cout << "Deformation map" << std::endl;
    cv::Mat id_minus_svf = identity_map - sampled_svf;
    performScalingAndSquaring(deformation_map,id_minus_svf,N);
    
    std::cout << "Done" << std::endl;
}

void create_mask_edge_object(cv::Mat& edge_map, const cv::Mat& label_image, unsigned char value_of_interest)
{
    cv::Mat mask_object, eroded_mask_object, eroded_mask_object_comp;
    create_mask_object(mask_object,label_image,value_of_interest);
    cv::Mat struct_el = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
    cv::erode(mask_object,eroded_mask_object,struct_el);
   
    eroded_mask_object_comp = 255 - eroded_mask_object;
    edge_map = eroded_mask_object_comp.mul(mask_object);
 
 //   cv::morphologyEx(input_mask,closed_mask,cv::MORPH_CLOSE,struct_el);
 //   cv::imshow("Edge map",edge_map);
 //   cv::waitKey(-1);

    
}


void load_true_transformation_from_deformation_map(std::map<cv::Point2f,cv::Point2f,comp_Point2f>& true_transformation, const cv::Mat& deformation_map, const cv::Mat& mask, unsigned char value_of_interest)
{
    CV_Assert(mask.type()==CV_8UC1);
    CV_Assert(deformation_map.type()==CV_32FC2);
    
    true_transformation.clear();
    
    int dimx = mask.cols;
    int dimy = mask.rows;
    for (int y=0; y<dimy; ++y)
    {
        const float *row_ptr = deformation_map.ptr<float>(y);
        for (int x=0; x<dimx; ++x)
        {
            if (mask.at<unsigned char>(y,x)==value_of_interest)
            {
                float im_x = row_ptr[2*x];
                float im_y = row_ptr[2*x+1];
                cv::Point2f input_point(x,y), output_point(im_x,im_y);
                true_transformation[input_point] = output_point;
            }
        }
    }
    
}

void load_locations_of_interest_from_mask(std::set<cv::Point2f,comp_Point2f>& locations, const cv::Mat& mask, unsigned char value_of_interest, int subsampling)
{
    CV_Assert(mask.type()==CV_8UC1);
    locations.clear();
    
    int dimx = mask.cols;
    int dimy = mask.rows;
    for (int y=0; y<dimy; y+=subsampling)
    {
        for (int x=0; x<dimx; x+=subsampling)
        {
            if (mask.at<unsigned char>(y,x)==value_of_interest)
                locations.insert(cv::Point2f(x,y));
        }
    }
    
}



void create_mask_object(cv::Mat& mask_object, const cv::Mat& label_image, unsigned char value_of_interest)
{
    CV_Assert(label_image.type()==CV_8UC1);
    
    int dimx = label_image.cols;
    int dimy = label_image.rows;
    mask_object = cv::Mat(dimy,dimx,CV_8UC1,cv::Scalar::all(0));
    std::cout << dimx << " " << dimy << std::endl;
    for (int y=0; y<dimy; ++y)
    {
        for (int x=0; x<dimx; ++x)
        {
            if (label_image.at<unsigned char>(y,x)==value_of_interest)
                mask_object.at<unsigned char>(y,x) = 255;
        }
    }
}


void load_landmark_file(std::vector<Eigen::Vector2d>& loaded_landmarks, const std::string& csv_filename)
{
    loaded_landmarks.clear();
    std::string read_str;
    std::ifstream infile(csv_filename);
    int col(-2), x, y;
    while (!infile.eof())
    {
        std::getline(infile,read_str,',');
        if (col==1)
            x = std::stoi(read_str);
        if (col==2)
            y = std::stoi(read_str);
        ++col;
        if (col==3)
        {
            loaded_landmarks.push_back(Eigen::Vector2d(x,y));
            col = 1;
        }
    }
    infile.close();
}

void load_true_transformation_from_landmark_files(std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& true_transformation, const std::string& landmarks_fixed_image_filename, const std::string& landmarks_moving_image_filename)
{
    // Load corresponding ground truth transformation
    true_transformation.clear();
    std::vector<Eigen::Vector2d> landmarks_fixed_image, landmarks_moving_image;
    load_landmark_file(landmarks_fixed_image,landmarks_fixed_image_filename);
    load_landmark_file(landmarks_moving_image,landmarks_moving_image_filename);
    int nb_gt_landmarks = landmarks_fixed_image.size();
    for (int l=0; l<nb_gt_landmarks; ++l)
        true_transformation[landmarks_fixed_image[l]] = landmarks_moving_image[l];
}





void convert_from_cv_to_eigen(std::set<Eigen::Vector2d,comp_Point<2>>& set_eigen, const std::set<cv::Point2f,comp_Point2f>& set_cv) 
{
    set_eigen.clear();
    for (const auto& p_cv : set_cv) 
    {
        Eigen::Vector2d p_eigen(p_cv.x, p_cv.y);
        set_eigen.insert(p_eigen);
    }
}


void convert_from_cv_to_eigen(std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& map_eigen, const std::map<cv::Point2f,cv::Point2f,comp_Point2f>& map_cv) 
{
    map_eigen.clear();
    for (const auto& p_cv : map_cv) 
    {
        Eigen::Vector2d p_eigen_first(p_cv.first.x, p_cv.first.y), p_eigen_second(p_cv.second.x, p_cv.second.y);
        map_eigen.insert(std::pair<Eigen::Vector2d,Eigen::Vector2d>(p_eigen_first,p_eigen_second));
    }
}

void convert_from_cv_to_eigen(std::vector<std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>>& vector_map_eigen, const std::vector<std::map<cv::Point2f,cv::Point2f,comp_Point2f>>& vector_map_cv) 
{
    int N = (int)vector_map_cv.size();
    vector_map_eigen.resize(N);
    for (int k=0; k<N; ++k)
        convert_from_cv_to_eigen(vector_map_eigen[k],vector_map_cv[k]);
}



void read_transformix_output_points(cv::Mat& transformation, const std::string& output_points_filename)
{
    CV_Assert(transformation.type()==CV_32FC2);
    
    std::string line;
    std::ifstream infile(output_points_filename);
    int l(0);
    std::string temp_str;
    while (std::getline(infile,line,';'))
    {
        int input_point_x, input_point_y;
        float output_point_x, output_point_y;
        std::stringstream s_stream(line);
        if ((l % 5)==1)
        {
            for (int k=0; k<3; ++k)
                s_stream >> temp_str;
            s_stream >> input_point_x;
            s_stream >> input_point_y;
        }
        if ((l % 5)==4)
        {
            for (int k=0; k<3; ++k)
                s_stream >> temp_str;

            s_stream >> output_point_x;
            s_stream >> output_point_y;
            float *row_ptr = transformation.ptr<float>(input_point_y);
            row_ptr[(int)2*input_point_x] = output_point_x;
            row_ptr[(int)2*input_point_x + 1] = output_point_y;
        }

     //   std::cout << line << std::endl;
        ++l;
    }
    
    infile.close();
}


void register_and_transform_with_elastix(std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>& transformation, std::set<Eigen::Vector2d,comp_Point<2>>& target_points, const cv::Mat& fixed_image, const cv::Mat& moving_image, const std::string& pair_identifier, const std::string& temp_folder, const std::string& elastix_registration_parameters, bool load_precomputed_file, bool remove_created_files, CoordinateType coordinate_type)
{
    std::string folder_registration = temp_folder + "/Registration-" + pair_identifier;
    std::string cmd_create_directory = "mkdir " + folder_registration;
    int res_command;
    res_command = system(cmd_create_directory.c_str());
    
    std::string fixed_image_filename = folder_registration + "/fixed_image.png";
    std::string moving_image_filename = folder_registration + "/moving_image.png";
    
    if (!load_precomputed_file)
    {
        cv::imwrite(fixed_image_filename,fixed_image);
        cv::imwrite(moving_image_filename,moving_image);
    }
    
    register_and_transform_with_elastix(transformation,target_points,fixed_image_filename,moving_image_filename,folder_registration,elastix_registration_parameters,load_precomputed_file,coordinate_type);
    
    // Remove the created files
    if ((!load_precomputed_file) && (remove_created_files))
    {
        std::string cmd_remove_files = "rm " + fixed_image_filename + " " + moving_image_filename + " " + folder_registration + "/result.0.png";
        res_command = system(cmd_remove_files.c_str());
    }
}

void tranform_color_image_with_elastix(const std::string& output_image_filename, const std::string& input_image_filename, const std::string& elastix_transformation_file, const std::string& temp_folder)
{
    cv::Mat input_image = cv::imread(input_image_filename,CV_LOAD_IMAGE_COLOR);
    
    std::string input_image_prefix = removeExtension(input_image_filename);
    
    // Split the channels
    std::vector<cv::Mat> image_channels(3), transformed_image_channels(3);
    cv::split(input_image,image_channels);

    // Transform and stored each channel independently
    for (int c=0; c<3; ++c)
    {
        std::ostringstream oss_c;
        oss_c << c;
        std::string image_channel_name = input_image_prefix + "-Channel" + oss_c.str() + ".png";
        
        // Save the channel image
        cv::imwrite(image_channel_name,image_channels[c]);
        
        // Transform it
        std::string cmd_transform = "transformix -in " + image_channel_name + " -out " + temp_folder + " -tp " + elastix_transformation_file + " >/dev/null";
        int res_command;
        res_command = system(cmd_transform.c_str());
        
        // Load the result
        transformed_image_channels[c] = cv::imread(temp_folder + "/result.png",CV_LOAD_IMAGE_GRAYSCALE);
    }

    //Merge the registered channels
    cv::Mat transformed_image;
    cv::merge(transformed_image_channels,transformed_image);
   
    // Finally, write it
    cv::imwrite(output_image_filename,transformed_image);
    
}

