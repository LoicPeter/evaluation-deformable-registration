
#include "processing_scripts_3d.hpp"
#include <itkNiftiImageIO.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>

void square_exponential_vector_field(BufferImage3D<Eigen::Vector3d>& output_map, BufferImage3D<Eigen::Vector3d>& input_map)
{
    int dimx = input_map.getDimension(0);
    int dimy = input_map.getDimension(1);
    int dimz = input_map.getDimension(2);
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                Eigen::Vector3d target = input_map.at(x,y,z);  // Target location at which we have to interpolate
                
                if ((target(0)<0) || (target(0)>(dimx-1)) || (target(1)<0) || (target(1)>(dimy-1)) || (target(2)<0) || (target(2)>(dimz-1)))
                    output_map.at(x,y,z) = target;
                else
                    output_map.at(x,y,z) = input_map.interpolate(target(0),target(1),target(2));
            }
            
        }
    }
}


void perform_scaling_and_squaring(BufferImage3D<Eigen::Vector3d>& output_map, const BufferImage3D<Eigen::Vector3d>& SVF_displacement, int N)
{
    int dimx = SVF_displacement.getDimension(0);
    int dimy = SVF_displacement.getDimension(1);
    int dimz = SVF_displacement.getDimension(2);
    
    BufferImage3D<Eigen::Vector3d> initial_map(dimx,dimy,dimz); // we add the identity
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                initial_map.at(x,y,z) = Eigen::Vector3d(x,y,z) + (1/(std::pow<double>(2,N)))*(SVF_displacement.at(x,y,z));
            }
        }
    }

    for (int k=0; k<N; ++k)
    {
        square_exponential_vector_field(output_map,initial_map);
        if (k!=(N-1))
            initial_map = output_map;
    }
}

// Returns the parameters mu_uni and sigma_square_uni such that X1, X2 and X3 follow LN(mu_uni,sigma_square_uni). These parameters are estimated such that the L2 norm of (X1, X2, X3) has mean original_mean_error and variance original_variance_error
void get_unidimensional_error_lognormal_parameters(double& mu_uni, double& sigma_square_uni, double original_mean_error, double original_variance_error)
{
    // First, define the corresponding mu and sigma_square followed by the L2 norm underlognormal assumption
    double squared_original_mean_error =  std::pow<double>(original_mean_error,2);
    double sigma_square = std::log(1 + original_variance_error/squared_original_mean_error);
    double mu = std::log(original_mean_error) - 0.5*sigma_square;
    
    // Now compute the desired parameters (Fenton-Wilkinson approximation)
    sigma_square_uni = 0.25*std::log(3*std::exp(4*sigma_square) - 2);
    mu_uni = mu + 0.25*std::log(2 + std::exp(4*sigma_square_uni)) - sigma_square_uni - 0.75*std::log(3);
}

void load_true_transformation_dirlab(std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>& true_transformation, const std::string& fixed_image_filename, const std::string& moving_image_filename, const Eigen::Vector3d& spacing)
{
    std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>> true_transformation_in_pixels;
    double a, b, c, d, e, f;
	std::ifstream infile_fixed_image(fixed_image_filename), infile_moving_image(moving_image_filename);
    while ((infile_fixed_image >> a >> b >> c) && (infile_moving_image >> d >> e >> f))
    {
        Eigen::Vector3d landmark_fixed_image(b,a,c);
        Eigen::Vector3d landmark_moving_image(e,d,f);
        true_transformation_in_pixels.insert(std::pair<Eigen::Vector3d,Eigen::Vector3d>(landmark_fixed_image,landmark_moving_image));
    }
    
    std::cout << "In load_true_transformation_dirlab: we need to get read of apply_spacing and use the conversion to physical points instead" << std::endl;
    apply_spacing<3>(true_transformation,true_transformation_in_pixels,spacing);
}

void extract_slice_from_buffer(cv::Mat& slice, const BufferImage3D<unsigned char>& input_buffer, int coordinate, int coordinate_value)
{
    int dimx = input_buffer.getDimension(0);
    int dimy = input_buffer.getDimension(1);
    int dimz = input_buffer.getDimension(2);
    
    std::cout << "Dimensions of input image: " << dimx << " " << dimy << " " << dimz << std::endl;
    
    int nb_rows(0), nb_cols(0);
    if (coordinate==0) // if we extract the slice x = coordinate_value
    {
        nb_rows = dimy;
        nb_cols = dimz;
        slice = cv::Mat(nb_rows, nb_cols, CV_8UC1, cv::Scalar::all(0));
        for (int y=0; y<nb_rows; ++y)
        {
            for (int z=0; z<nb_cols; ++z)
                slice.at<unsigned char>(y,z) = input_buffer.at(coordinate_value,y,z);
        }
    }
    
    if (coordinate==1) // if we extract the slice y = coordinate_value
    {
        nb_rows = dimz;
        nb_cols = dimx;
        slice = cv::Mat(nb_rows, nb_cols, CV_8UC1, cv::Scalar::all(0));
        for (int z=0; z<nb_rows; ++z)
        {
            for (int x=0; x<nb_cols; ++x)
                slice.at<unsigned char>(z,x) = input_buffer.at(x,coordinate_value,z);
        }
    }
    
    if (coordinate==2) // if we extract the slice z = coordinate_value
    {
        nb_rows = dimy;
        nb_cols = dimx;
        slice = cv::Mat(nb_rows, nb_cols, CV_8UC1, cv::Scalar::all(0));
        for (int y=0; y<nb_rows; ++y)
        {
            for (int x=0; x<nb_cols; ++x)
                slice.at<unsigned char>(y,x) = input_buffer.at(x,y,coordinate_value);
        }
    }
    
    std::cout << "Extraction done" << std::endl;
    
}



