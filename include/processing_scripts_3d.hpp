#ifndef PROCESSING_SCRIPTS_3D_INCLUDED
#define PROCESSING_SCRIPTS_3D_INCLUDED

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
//#include "utilities.hpp"

#include <omp.h>

//#include "registration_averaging.hpp"
#include "landmark_suggestion.hpp"
#include "mean_function.hpp"
#include "gaussian_kernel.hpp"
#include "wendland_kernel.hpp"
#include "gaussian_process.hpp"
#include "processing_scripts.hpp"
#include "visualization_functions.hpp"
#include "buffer_image_3d.hpp"

// For solving system
#include "stdafx.h"
#include "optimization.h"

#include <itkImageBase.h>
#include <itkImage.h>
#include <itkImageDuplicator.h>
#include <itkNiftiImageIO.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkSmartPointer.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryMorphologicalClosingImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkSobelEdgeDetectionImageFilter.h>
#include <itkGradientImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkPoint.h>

// Random number / probabilities / sampling
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

template <typename CoordinateType, typename ImageType>
itk::Index<3> physicalPointToIndex(const itk::Point<CoordinateType,3>& physical_point, const itk::SmartPointer<ImageType> image)
{
    itk::Index<3> coordinates;
    image->TransformPhysicalPointToIndex(physical_point,coordinates);
    return coordinates;
}

template <typename CoordinateType, typename ImageType>
itk::Point<CoordinateType,3> indexToPhysicalPoint(const itk::Index<3>& coordinates, const itk::SmartPointer<ImageType> image)
{
    itk::Point<CoordinateType,3> physical_point;
    image->TransformIndexToPhysicalPoint(coordinates,physical_point);
    return physical_point;
}

template <typename CoordinateType, typename ImageType>
Eigen::Matrix<CoordinateType,3,1> indexToPhysicalPointEigen(const itk::Index<3>& coordinates, const itk::SmartPointer<ImageType> image)
{
    itk::Point<CoordinateType,3> physical_point = indexToPhysicalPoint<CoordinateType,ImageType>(coordinates,image);
    return Eigen::Matrix<CoordinateType,3,1>(physical_point[0],physical_point[1],physical_point[2]);
}

template <typename PixelType>
itk::SmartPointer<itk::Image<PixelType,3>> read_nifti_file(const std::string& filename)
{
    using ReaderType = itk::ImageFileReader<itk::Image<PixelType,3>>;
    typename ReaderType::Pointer reader = ReaderType::New();
    itk::NiftiImageIO::Pointer IOPtr= itk::NiftiImageIO::New();
    reader->SetImageIO(IOPtr);
    reader->SetFileName(filename);
    reader->Update();
    return reader->GetOutput(); 
}

template <typename PixelType>
void write_as_nifti_file(const std::string& filename, const itk::SmartPointer<itk::Image<PixelType,3>> input_image)
{
    using WriterType = itk::ImageFileWriter<itk::Image<PixelType,3>>;
    typename WriterType::Pointer writer = WriterType::New();
    itk::NiftiImageIO::Pointer IOPtr= itk::NiftiImageIO::New();
    writer->SetImageIO(IOPtr);
    writer->SetInput(input_image);
    writer->SetFileName(filename);
    writer->Update();
}

void square_exponential_vector_field(BufferImage3D<Eigen::Vector3d>& output_map, BufferImage3D<Eigen::Vector3d>& input_map);

void perform_scaling_and_squaring(BufferImage3D<Eigen::Vector3d>& output_map, const BufferImage3D<Eigen::Vector3d>& SVF, int N);

template <typename PixelType>
void get_size(int& dimx, int& dimy, int& dimz, const itk::SmartPointer<itk::Image<PixelType,3>> input_image)
{
    using ImageType = itk::Image<PixelType,3>;
    typename ImageType::RegionType region = input_image->GetLargestPossibleRegion();
    typename ImageType::SizeType imsize = region.GetSize();
    dimx = imsize[0];
    dimy = imsize[1];
    dimz = imsize[2];
    
}

template <typename PixelType>
void applyDeformationImage(BufferImage3D<PixelType>& output_image, const BufferImage3D<PixelType>& input_image, const BufferImage3D<Eigen::Vector3d>& input_map)
{
    int dimx = input_image.getDimension(0);
    int dimy = input_image.getDimension(1);
    int dimz = input_image.getDimension(2);
    
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                Eigen::Vector3d target = input_map.at(x,y,z);
                output_image.at(x,y,z) = input_image.interpolate(target(0),target(1),target(2));
            }
        }
    }
}

// All sizes assumed correct
template <typename PixelType>
void itk_to_buffer(BufferImage3D<PixelType>& output_buffer, const itk::SmartPointer<itk::Image<PixelType,3>>& input_itk_image)
{
    int dimx = output_buffer.getDimension(0);
    int dimy = output_buffer.getDimension(1);
    int dimz = output_buffer.getDimension(2);
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                output_buffer.at(x,y,z) = input_itk_image->GetPixel(coordinates);
            }
        }
    }
}

template <typename PixelType>
void buffer_to_itk(itk::SmartPointer<itk::Image<PixelType,3>>& output_itk_image, const BufferImage3D<PixelType>& input_buffer)
{
    int dimx = input_buffer.getDimension(0);
    int dimy = input_buffer.getDimension(1);
    int dimz = input_buffer.getDimension(2);
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                output_itk_image->SetPixel(coordinates,input_buffer.at(x,y,z));
            }
        }
    }
}

template <typename PixelType>
void detect_salient_locations(std::set<Eigen::Vector3d,comp_Point<3>>& locations, const itk::SmartPointer<itk::Image<PixelType,3>>& input_image, const itk::SmartPointer<itk::Image<unsigned char,3>> input_mask)
{
    using ImageType = itk::Image<PixelType,3>;
    using GradientType = itk::Image<float,3>;

    locations.clear();
    typename itk::GradientImageFilter< ImageType, float>::Pointer gradientFilter = itk::GradientImageFilter< ImageType, float>::New();
    gradientFilter->SetInput(input_image);
    gradientFilter->Update();
    
    int dimx, dimy, dimz;
    get_size(dimx, dimy, dimz, input_image);
    
    int window_half_size = 3;
    double K = 0;
    double response_threshold = 0.0001;
    itk::Image<itk::CovariantVector<float,9>>::Pointer hessian_image = itk::Image<itk::CovariantVector<float,9>>::New();
    BufferImage3D<float> corner_response(dimx,dimy,dimz,0);
    for (int z=window_half_size; z<(dimz-window_half_size); ++z)
    {
        for (int y=window_half_size; y<(dimy-window_half_size); ++y)
        {
            for (int x=window_half_size; x<(dimx-window_half_size); ++x)
            {
                Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
                bool isInsideMask(true);
                for (int dz=-window_half_size; dz<=window_half_size; ++dz)
                {
                      for (int dy=-window_half_size; dy<=window_half_size; ++dy)
                      {
                           for (int dx=-window_half_size; dx<=window_half_size; ++dx)
                           {
                                itk::Index<3> coordinates;
                                coordinates[0] = x + dx;
                                coordinates[1] = y + dy;
                                coordinates[2] = z + dz;
                                if (input_mask->GetPixel(coordinates)==255)
                                {
                                    itk::CovariantVector<float,3> g = gradientFilter->GetOutput()->GetPixel(coordinates);
                                    Eigen::Vector3d g_eig(g[0],g[1],g[2]);
                                    M += g_eig*(g_eig.transpose());
                                }
                                else
                                    isInsideMask = false;
                           }
                      }
                }
                
                if (isInsideMask)
                {
                    corner_response.at(x,y,z) = M.determinant() - K*std::pow<double>(M.trace(),3);
                    //std::cout << M << std::endl;
                    //std::cout << M.determinant() << std::endl;
                    //corner_response.at(x,y,z) = M.trace();
                    //std::cout << corner_response.at(x,y,z) << " ";
                  /*  
                    itk::Index<3> coordinates;
                    coordinates[0] = x;
                    coordinates[1] = y;
                    coordinates[2] = z;
                    itk::CovariantVector<float,3> g = gradientFilter->GetOutput()->GetPixel(coordinates);
                    Eigen::Vector3d g_eig(g[0],g[1],g[2]);
                    corner_response.at(x,y,z) = (float)g_eig.norm();*/
                }
                else
                    corner_response.at(x,y,z) = 0;
            }
        }
    }
    

    // Deep copy of input image (allows to keep meta information intact and we just fill the buffer)
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
    duplicator->SetInputImage(input_image);
    duplicator->Update();
    typename ImageType::Pointer visualisation_image = duplicator->GetOutput();
    
    // Visualisation of Harris response for debug purposes
//     typename GradientType::Pointer corner_response_itk = GradientType::New();
//     GradientType::IndexType responsestart;
//     responsestart[0] = 0; // first index on X
//     responsestart[1] = 0; // first index on Y
//     responsestart[2] = 0; // first index on Z
//     GradientType::SizeType responsesize;
//     responsesize[0] = dimx; // size along X
//     responsesize[1] = dimy; // size along Y
//     responsesize[2] = dimz; // size along Z
//     GradientType::RegionType responseregion;
//     responseregion.SetSize(responsesize);
//     responseregion.SetIndex(responsestart);
//     corner_response_itk->SetRegions(responseregion);
//     corner_response_itk->Allocate();
//     corner_response_itk->SetOrigin(input_image->GetOrigin());
//     corner_response_itk->SetDirection(input_image->GetDirection());
//     corner_response_itk->SetSpacing(input_image->GetSpacing());
//     buffer_to_itk(corner_response_itk,corner_response);
  //  write_as_nifti_file<float>("Harris_response.nii",corner_response_itk);
    
    
    std::vector<Eigen::Vector3i> local_maxima;
    extract_maxima(local_maxima,corner_response);
    for (const Eigen::Vector3i& maxPt : local_maxima)
    {
        itk::Index<3> coordinates;
        coordinates[0] = maxPt(0);
        coordinates[1] = maxPt(1);
        coordinates[2] = maxPt(2);
        if (corner_response.at(coordinates[0],coordinates[1],coordinates[2])>response_threshold)
        {

            Eigen::Vector3d cornerPoint = indexToPhysicalPointEigen<double,ImageType>(coordinates,input_image); 
            locations.insert(cornerPoint);
            visualisation_image->SetPixel(coordinates,255);
        }
        else
            visualisation_image->SetPixel(coordinates,0);
    }
    
    write_as_nifti_file<PixelType>("Corners.nii",visualisation_image);

}

template <typename LabelType>
void extract_unique_labels(std::set<LabelType>& labels, const itk::SmartPointer<itk::Image<LabelType,3>> label_image)
{
    labels.clear();
    int dimx, dimy, dimz;
    get_size(dimx, dimy, dimz, label_image);
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                LabelType val = label_image->GetPixel(coordinates);
                if (labels.count(val)==0)
                    labels.insert(val);
            }
        }
    }
}

void extract_slice_from_buffer(cv::Mat& slice, const BufferImage3D<unsigned char>& input_buffer, int coordinate, int coordinate_value);


template <typename PixelType>
void extract_locations_in_slices(std::set<Eigen::Vector2d,comp_Point<2>>& locations_in_x, std::set<Eigen::Vector2d,comp_Point<2>>& locations_in_y, std::set<Eigen::Vector2d,comp_Point<2>>& locations_in_z, const std::set<Eigen::Vector3d,comp_Point<3>>& input_locations, const itk::SmartPointer<itk::Image<PixelType,3>>& input_image, int x_slice, int y_slice, int z_slice)
{
    locations_in_x.clear();
    locations_in_y.clear();
    locations_in_z.clear();
    
    for (const Eigen::Vector3d& p_3d: input_locations)
    {
        itk::Point<double,3> p_itk;
        p_itk[0] = p_3d(0);
        p_itk[1] = p_3d(1);
        p_itk[2] = p_3d(2);
       
        itk::Index<3> coordinates = physicalPointToIndex(p_itk,input_image);
        if (coordinates[0]==x_slice)
            locations_in_x.insert(Eigen::Vector2d(coordinates[2], coordinates[1]));
        if (coordinates[1]==y_slice)
            locations_in_y.insert(Eigen::Vector2d(coordinates[0], coordinates[2]));
        if (coordinates[2]==z_slice)
            locations_in_z.insert(Eigen::Vector2d(coordinates[0], coordinates[1]));
    }
    
}


template <typename PixelType>
void plot_salient_and_target_locations(const itk::SmartPointer<itk::Image<PixelType,3>>& fixed_image, const itk::SmartPointer<itk::Image<PixelType,3>>& moving_image, const std::set<Eigen::Vector3d,comp_Point<3>>& salient_locations, const std::set<Eigen::Vector3d,comp_Point<3>>& target_locations, int x, int y, int z)
{
    int dimx_fixed_image, dimy_fixed_image, dimz_fixed_image, dimx_moving_image, dimy_moving_image, dimz_moving_image;
    get_size(dimx_fixed_image, dimy_fixed_image, dimz_fixed_image, fixed_image);
    get_size(dimx_moving_image, dimy_moving_image, dimz_moving_image, moving_image);
    
    BufferImage3D<PixelType> fixed_image_buffer(dimx_fixed_image,dimy_fixed_image,dimz_fixed_image), moving_image_buffer(dimx_moving_image,dimy_moving_image,dimz_moving_image);
    itk_to_buffer(fixed_image_buffer,fixed_image);
    itk_to_buffer(moving_image_buffer,moving_image);

    std::set<Eigen::Vector2d,comp_Point<2>> salient_in_x, salient_in_y, salient_in_z, target_in_x, target_in_y, target_in_z;
    extract_locations_in_slices(salient_in_x, salient_in_y, salient_in_z, salient_locations, fixed_image, x, y, z);
    extract_locations_in_slices(target_in_x, target_in_y, target_in_z, target_locations, fixed_image, x, y, z);
    
    cv::Mat x_slice_fixed, y_slice_fixed, z_slice_fixed;
    extract_slice_from_buffer(x_slice_fixed,fixed_image_buffer,0,x);
    extract_slice_from_buffer(y_slice_fixed,fixed_image_buffer,1,y);
    extract_slice_from_buffer(z_slice_fixed,fixed_image_buffer,2,z);
    
    std::cout << "Plot salient and target points" << std::endl;
    plot_salient_and_target_points(x_slice_fixed, salient_in_x, target_in_x, "X-Slice-Fixed.png");
    plot_salient_and_target_points(y_slice_fixed, salient_in_y, target_in_y, "Y-Slice-Fixed.png");
    plot_salient_and_target_points(z_slice_fixed, salient_in_z, target_in_z, "Z-Slice-Fixed.png");

    std::cout << "Extract moving image slices" << std::endl;
    cv::Mat x_slice_moving, y_slice_moving, z_slice_moving;
    extract_slice_from_buffer(x_slice_moving,moving_image_buffer,0,x);
    extract_slice_from_buffer(y_slice_moving,moving_image_buffer,1,y);
    extract_slice_from_buffer(z_slice_moving,moving_image_buffer,2,z);
    
    normalise_and_write(x_slice_moving, "X-Slice-Moving.png");
    normalise_and_write(y_slice_moving, "Y-Slice-Moving.png");
    normalise_and_write(z_slice_moving, "Z-Slice-Moving.png");
}

template <typename LabelType>
void convert_binary_mask_for_elastix(itk::SmartPointer<itk::Image<LabelType,3>>& output_mask, const itk::SmartPointer<itk::Image<LabelType,3>> input_mask)
{
    using LabelImageType = itk::Image<LabelType,3>;
    typename itk::BinaryThresholdImageFilter<LabelImageType,LabelImageType>::Pointer threshold_filter = itk::BinaryThresholdImageFilter<LabelImageType,LabelImageType>::New();
    threshold_filter->SetInput(input_mask);
    threshold_filter->SetLowerThreshold(1);
    threshold_filter->SetUpperThreshold(255);
    threshold_filter->SetInsideValue(1);
    threshold_filter->Update();
    output_mask = threshold_filter->GetOutput();
}

// Creates deformed_image and deformation_map such that  "deformed_image(deformation_map(x)) = input_image(x)"
template <typename PixelType>
void create_synthetic_transformation(itk::SmartPointer<itk::Image<PixelType,3>>& deformed_image, BufferImage3D<Eigen::Vector3d>& deformation_map, const itk::SmartPointer<itk::Image<PixelType,3>>& input_image, const itk::SmartPointer<itk::Image<unsigned char,3>> input_mask, double std_gaussian, double max_svf, double reduction_factor, boost::random::mt19937& rng, int nb_scaling_squaring = 10)
{
    using ImageType = itk::Image<PixelType,3>;
    using MaskType = itk::Image<unsigned char,3>;

    typename ImageType::RegionType region = input_image->GetLargestPossibleRegion();
    typename ImageType::SizeType size_image = region.GetSize(); 
    int dimx = size_image[0];
    int dimy = size_image[1];
    int dimz = size_image[2];
    
    // Build the structuring element for the erosion (to keep edges of the brain undeformed)   
    itk::BinaryBallStructuringElement<unsigned char, 3> struct_el;
    struct_el.SetRadius(ceil(std_gaussian));
    struct_el.CreateStructuringElement();
    
    // Erode and smooth the mask
    typename itk::BinaryMorphologicalClosingImageFilter<MaskType,MaskType,itk::BinaryBallStructuringElement<unsigned char, 3>>::Pointer erosion_filter = itk::BinaryMorphologicalClosingImageFilter<ImageType,ImageType,itk::BinaryBallStructuringElement<unsigned char, 3>>::New();
    typename itk::DiscreteGaussianImageFilter<MaskType,MaskType>::Pointer smoothing_filter =  itk::DiscreteGaussianImageFilter<MaskType,MaskType>::New();
    erosion_filter->SetForegroundValue(255);
    erosion_filter->SetKernel(struct_el);
    erosion_filter->SetInput(input_mask);
    smoothing_filter->SetUseImageSpacingOff();
    smoothing_filter->SetVariance(std_gaussian*std_gaussian);
    smoothing_filter->SetInput(erosion_filter->GetOutput());
    smoothing_filter->Update();
    MaskType::Pointer smooth_mask = smoothing_filter->GetOutput();

    // Create a small image where each pixel is sampled independently
    boost::uniform_real<float> svf_distribution(-max_svf,max_svf); // Distribution on SVF
    int dimx_downsized = std::round(reduction_factor*dimx);
    int dimy_downsized = std::round(reduction_factor*dimy);
    int dimz_downsized = std::round(reduction_factor*dimz);
    BufferImage3D<Eigen::Vector3d> downsized_svf(dimx_downsized,dimy_downsized,dimz_downsized), sampled_svf(dimx,dimy,dimz);
    for (int z=0; z<dimz_downsized; ++z)
    {
        for (int y=0; y<dimy_downsized; ++y)
        {
            for (int x=0; x<dimx_downsized; ++x)
            {
                double dx = svf_distribution(rng);
                double dy = svf_distribution(rng);
                double dz = svf_distribution(rng);
                Eigen::Vector3d disp(dx,dy,dz);
                downsized_svf.at(x,y,z) = disp;
            }
        }
    }
    
    resize(sampled_svf,downsized_svf);

    
    // We multiply the svf with the mask
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                itk::Point<double,3> physicalPoint = indexToPhysicalPoint<double,ImageType>(coordinates,input_image);
                itk::Index<3> coordinatesInMask = physicalPointToIndex(physicalPoint,smooth_mask);
                double smoothMaskVal = ((double)smooth_mask->GetPixel(coordinatesInMask))/255.0;
                sampled_svf.at(x,y,z) *= smoothMaskVal;
            }
        }
    }
    
    // Deep copy of input image (allows to keep meta information intact and we just fill the buffer)
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
    duplicator->SetInputImage(input_image);
    duplicator->Update();
    deformed_image = duplicator->GetOutput();
    
    
    std::cout << "Scaling and squaring" << std::endl;
    BufferImage3D<Eigen::Vector3d> deformation_map_inv(dimx,dimy,dimz);
    perform_scaling_and_squaring(deformation_map_inv,sampled_svf,nb_scaling_squaring);
    
//     for (int z=0; z<dimz; ++z)
//     {
//         for (int y=0; y<dimy; ++y)
//         {
//             for (int x=0; x<dimx; ++x)
//             {
//                 itk::Index<3> coordinates;
//                 coordinates[0] = x;
//                 coordinates[1] = y;
//                 coordinates[2] = z;
//                 double smoothMaskVal = ((double)smooth_mask->GetPixel(coordinates))/255.0;
//                 if (smoothMaskVal!=0)
//                 {
//                     std::cout << "(" << x << "," << y << "," << z << "): " << smoothMaskVal << " / " << deformation_map_inv.at(x,y,z).transpose() << std::endl;
//                 }
//                     
//             }
//         }
//     }

    
    std::cout << "Apply inverse deformation field to image" << std::endl;
    BufferImage3D<PixelType> input_image_buffer(dimx,dimy,dimz), deformed_image_buffer(dimx,dimy,dimz);
    itk_to_buffer(input_image_buffer,input_image);
    applyDeformationImage(deformed_image_buffer,input_image_buffer,deformation_map_inv);
    buffer_to_itk(deformed_image,deformed_image_buffer);
    
    // Deformation map
    std::cout << "Compute deformation field" << std::endl;
    multiply_in_place(-1,sampled_svf); // we invert the transformation
    perform_scaling_and_squaring(deformation_map,sampled_svf,nb_scaling_squaring);
    
    std::cout << "Done" << std::endl;
}

template <typename LabelType>
void load_true_transformation_from_deformation_map(std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>& true_transformation, const BufferImage3D<Eigen::Vector3d>& deformation_map, const itk::SmartPointer<itk::Image<LabelType,3>> input_mask, LabelType value_of_interest)
{

    true_transformation.clear();
    using ImageType = itk::Image<LabelType,3>;
    typename ImageType::RegionType region = input_mask->GetLargestPossibleRegion();
    typename ImageType::SizeType imsize = region.GetSize();
    typename ImageType::SpacingType imspacing = input_mask->GetSpacing();
    
    // Workaround to be able to interpolate the physical location out of the deformation map. The deformation map should probably be in physical space from the start instead...
    int dimx_buffer = deformation_map.getDimension(0);
    int dimy_buffer = deformation_map.getDimension(1);
    int dimz_buffer = deformation_map.getDimension(2);
    BufferImage3D<Eigen::Vector3d> deformation_map_physical_locations(dimx_buffer,dimy_buffer,dimz_buffer);
    for (int z=0; z<dimz_buffer; ++z)
    {
        for (int y=0; y<dimy_buffer; ++y)
        {
            for (int x=0; x<dimx_buffer; ++x)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                deformation_map_physical_locations.at(x,y,z) = indexToPhysicalPointEigen<double,ImageType>(coordinates,input_mask);
            }
        }
    }
    
    
    int dimx = imsize[0];
    int dimy = imsize[1];
    int dimz = imsize[2];
    for (int z=0; z<dimz; ++z)
    {
        for (int y=0; y<dimy; ++y)
        {
            for (int x=0; x<dimx; ++x)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                if (input_mask->GetPixel(coordinates)==value_of_interest)
                {
                    Eigen::Vector3d input_point = indexToPhysicalPointEigen<double,ImageType>(coordinates,input_mask);
                    Eigen::Vector3d output_index = deformation_map.at(x,y,z);
                    Eigen::Vector3d output_point = deformation_map_physical_locations.interpolate(output_index(0),output_index(1),output_index(2));
                    true_transformation[input_point] = output_point;
                    //std::cout << "Point in fixed: " << input_point << " Point in moving: " << output_point << std::endl;
                }
            }
        }
    }
    
}

template <typename LabelType> 
itk::SmartPointer<itk::Image<LabelType,3>> create_mask_edge_object(const itk::SmartPointer<itk::Image<LabelType,3>> label_image, LabelType value_of_interest)
{
    using LabelImageType = itk::Image<LabelType,3>;
    using EdgeImageType = itk::Image<float,3>;
    typename itk::BinaryThresholdImageFilter<LabelImageType,LabelImageType>::Pointer label_threshold_filter = itk::BinaryThresholdImageFilter<LabelImageType,LabelImageType>::New();
    label_threshold_filter->SetInput(label_image);
    label_threshold_filter->SetLowerThreshold(value_of_interest);
    label_threshold_filter->SetUpperThreshold(value_of_interest);
    label_threshold_filter->Update();
    
    typename itk::SobelEdgeDetectionImageFilter<LabelImageType,EdgeImageType>::Pointer edge_detection_filter = itk::SobelEdgeDetectionImageFilter<LabelImageType,EdgeImageType>::New();
    edge_detection_filter->SetInput(label_threshold_filter->GetOutput());
    edge_detection_filter->Update();
    
    typename itk::BinaryThresholdImageFilter<EdgeImageType,LabelImageType>::Pointer edge_threshold_filter = itk::BinaryThresholdImageFilter<EdgeImageType,LabelImageType>::New();
    edge_threshold_filter->SetInput(edge_detection_filter->GetOutput());
    edge_threshold_filter->SetLowerThreshold(1);
    
    edge_threshold_filter->Update();
    return edge_threshold_filter->GetOutput();
}

template <typename LabelType> 
void load_locations_of_interest_from_mask(std::set<Eigen::Vector3d,comp_Point<3>>& locations, const itk::SmartPointer<itk::Image<LabelType,3>> input_mask, LabelType value_of_interest, int subsampling = 1)
{
    locations.clear();
    
    using ImageType = itk::Image<LabelType,3>;
    typename ImageType::RegionType region = input_mask->GetLargestPossibleRegion();
    typename ImageType::SizeType imsize = region.GetSize();
    typename ImageType::SpacingType imspacing = input_mask->GetSpacing();
    int dimx = imsize[0];
    int dimy = imsize[1];
    int dimz = imsize[2];
    
    // Deep copy of input image (allows to keep meta information intact and we just fill the buffer)
    typename itk::ImageDuplicator<ImageType>::Pointer duplicator = itk::ImageDuplicator<ImageType>::New();
    duplicator->SetInputImage(input_mask);
    duplicator->Update();
    typename ImageType::Pointer visualisation_image = duplicator->GetOutput();
    
    for (int z=0; z<dimz; z+=subsampling)
    {
        for (int y=0; y<dimy; y+=subsampling)
        {
            for (int x=0; x<dimx; x+=subsampling)
            {
                itk::Index<3> coordinates;
                coordinates[0] = x;
                coordinates[1] = y;
                coordinates[2] = z;
                if (input_mask->GetPixel(coordinates)==value_of_interest)
                {
                    Eigen::Vector3d pt = indexToPhysicalPointEigen<double,ImageType>(coordinates,input_mask);
                    locations.insert(pt);
                    visualisation_image->SetPixel(coordinates,158);
                }
                else
                    visualisation_image->SetPixel(coordinates,64);
            }
        }
    }
    
   // write_as_nifti_file<LabelType>("Locations_of_interest.nii",visualisation_image);
    
}


template <typename PixelType, typename LabelType> 
void register_and_transform_with_elastix(std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>& transformation, std::set<Eigen::Vector3d,comp_Point<3>>& target_points, itk::SmartPointer<itk::Image<PixelType,3>> fixed_image, itk::SmartPointer<itk::Image<PixelType,3>> moving_image, const std::string& pair_identifier, const std::string& temp_folder, const std::string& elastix_registration_parameters, bool load_precomputed_file, bool remove_created_files, CoordinateType coordinate_type, itk::SmartPointer<itk::Image<LabelType,3>> mask_image = nullptr)
{
    using ImageType = itk::Image<PixelType,3>;
    using WriterType = itk::ImageFileWriter<ImageType>;
    
    std::string folder_registration = temp_folder + "/Registration-" + pair_identifier;
    std::string cmd_create_directory = "mkdir " + folder_registration;

    int res_command;
    res_command = system(cmd_create_directory.c_str());
    
    std::string fixed_image_filename = folder_registration + "/fixed_image.nii";
    std::string moving_image_filename = folder_registration + "/moving_image.nii";
    std::string mask_image_filename = folder_registration + "/mask_image.nii";
        
    if (!load_precomputed_file)
    {
        itk::NiftiImageIO::Pointer IOPtr = itk::NiftiImageIO::New();
        
        // Write fixed image
        write_as_nifti_file<PixelType>(fixed_image_filename, fixed_image);
        
        // Write moving image
        write_as_nifti_file<PixelType>(moving_image_filename, moving_image);

        // Write mask image
        if (mask_image != nullptr)
        {
            itk::SmartPointer<itk::Image<LabelType,3>> mask_image_zero_one = itk::Image<LabelType,3>::New();
            convert_binary_mask_for_elastix<LabelType>(mask_image_zero_one, mask_image);
            write_as_nifti_file<LabelType>(mask_image_filename, mask_image_zero_one);
        }
           
            
    }
    
    if (mask_image != nullptr)
        register_and_transform_with_elastix(transformation,target_points,fixed_image_filename,moving_image_filename,mask_image_filename,folder_registration,elastix_registration_parameters,load_precomputed_file,coordinate_type);
    else
        register_and_transform_with_elastix(transformation,target_points,fixed_image_filename,moving_image_filename,folder_registration,elastix_registration_parameters,load_precomputed_file,coordinate_type);
    
    // Remove the created files
    if ((!load_precomputed_file) && (remove_created_files))
    {
        std::string cmd_remove_files;
        if (mask_image != nullptr)
            cmd_remove_files = "rm " + fixed_image_filename + " " + moving_image_filename + " " + mask_image_filename + " " + folder_registration + "/result.0.nii";
        else
            cmd_remove_files = "rm " + fixed_image_filename + " " + moving_image_filename + " " + folder_registration + "/result.0.nii";
        res_command = system(cmd_remove_files.c_str());
    }
    
}

void get_unidimensional_error_lognormal_parameters(double& mu_uni, double& sigma_square_uni, double original_mean_error, double original_variance_error);

void load_true_transformation_dirlab(std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>& true_transformation, const std::string& filename_fixed_image, const std::string& filename_moving_image, const Eigen::Vector3d& spacing);

//void run_dirlab_evaluation();

void run_copd_evaluation();

void run_T1_T2_evaluation();

void run_cobralab_example();

#endif // REGISTRATION_AVERAGING_INCLUDED
