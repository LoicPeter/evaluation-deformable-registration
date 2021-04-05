
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
#include "human_landmark_annotator.hpp"
#include "mean_function.hpp"
#include "gaussian_kernel.hpp"
#include "wendland_kernel.hpp"
#include "gaussian_process.hpp"
#include "processing_scripts_2d.hpp"
#include "processing_scripts_3d.hpp"

// For solving system
#include "stdafx.h"
#include "optimization.h"


#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// Random number / probabilities / sampling
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>



void run_cima_example()
{
 
    // Annotation settings
    double default_radius = 1;
    double min_radius = 5;
    double max_radius = 100;
    double mean_log_eigenvalues = 2.8927; // learned from manual annotations
    double std_log_eigenvalues = 0.8295;
    double confidence_level = 0.01;
    bool human_annotator = false;
    double scale_user = 1;
    double pdl = 0;
    
    // Parameters
    bool display(false);
    bool create_dense_maps(false);
    bool remove_created_files = false;
    double initial_scale = 10000;
    int nb_wendland_levels = 9;
    int subsampling_dense_maps = 9;
    double min_kernel_radius = 10; // in pixels
    Eigen::Vector2d spacing(1.0, 1.0); // everything given in pixels
    CoordinateType coordinateType = CoordinateType::index;
    std::vector<std::string> elastix_parameter_files = {"elastixconfig0.txt","elastixconfig15.txt","elastixconfig23.txt"};
    int nb_queries_for_burn_in = 0;
    int annotation_budget = 50;
    bool load_precomputed_registrations = false;
    bool pretend_locations_are_certain = false;
    QueryingStrategy querying_strategy(entropy,false); // suggestion strategy
    
    // Window for annotation
    PartialWindow partial_window_fixed_image("Fixed image",800,900,0,0,100,10);
    PartialWindow partial_window_moving_image("Moving image",800,900,0,0,1000,10);
    
    // Folders
    std::string data_folder = "../data/example-cima";
    std::string results_folder = "../results/example-cima";
    std::string stored_registrations_folder = data_folder + "/precomputed-registrations";
    
    // Random number generator
    boost::random::mt19937 rng;

    // Pair on which the kernel hyperparameters are learned 
    std::string training_fixed_image_landmarks = data_folder + "/cima-dataset/lung-lesion_1/29-041-Izd2-w35-Cc10-5-les1.csv";
    std::string training_moving_image_landmarks = data_folder + "/cima-dataset/lung-lesion_1/29-041-Izd2-w35-CD31-3-les1.csv";
    
    // Testing pair
    std::string fixed_image_filename = data_folder + "/cima-dataset/lung-lesion_2/29-041-Izd2-w35-Cc10-5-les2.jpg";
    std::string moving_image_filename = data_folder + "/cima-dataset/lung-lesion_2/29-041-Izd2-w35-CD31-3-les2.jpg";
    std::string fixed_image_landmarks = data_folder + "/cima-dataset/lung-lesion_2/29-041-Izd2-w35-Cc10-5-les2.csv"; // not supposed to be known
    std::string moving_image_landmarks = data_folder + "/cima-dataset/lung-lesion_2/29-041-Izd2-w35-CD31-3-les2.csv";
    std::string pair_identifier = "Cc10-5_CD31-3-les2"; // this string characterizes the pair
    
    // -------------------------------------------------------
    // Create the annotations for learning hyperparameters
    // -------------------------------------------------------
    
    // Load true transformation
    std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>> true_transformation_training;
    load_true_transformation_from_landmark_files(true_transformation_training,training_fixed_image_landmarks,training_moving_image_landmarks);
                        
    // Extract set of annotated locations
    std::set<Eigen::Vector2d,comp_Point<2>> gt_locations_training = create_set_from_map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>(true_transformation_training);
                    
    // Generate annotations
    PrecomputedLandmarkAnnotator<2> annotator_training_pair;
    annotator_training_pair.createfromBimodalLognormalDistribution(gt_locations_training,true_transformation_training,default_radius,confidence_level,mean_log_eigenvalues+ std::log(scale_user),std_log_eigenvalues + std::log(scale_user),0,rng,pretend_locations_are_certain);
    std::vector<std::vector<Gaussian_Process_Observation<2>>> training_annotations(1);
    training_annotations[0] = annotator_training_pair.getAllAnnotations();

                
    // -------------------------------------------------------
    // Pair of interest
    // -------------------------------------------------------
    

    // Load images
    cv::Mat fixed_image = cv::imread(fixed_image_filename);
    cv::Mat moving_image = cv::imread(moving_image_filename);
                    
    // Load corresponding ground truth transformation
    std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>> true_transformation;
    load_true_transformation_from_landmark_files(true_transformation,fixed_image_landmarks,moving_image_landmarks);
    
    // Define salient, target and evaluation points
    std::set<Eigen::Vector2d,comp_Point<2>> gt_locations = create_set_from_map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>(true_transformation); // all locations where the true transformation is available
    std::set<Eigen::Vector2d,comp_Point<2>> salient_locations = gt_locations;
    std::set<Eigen::Vector2d,comp_Point<2>> target_locations = gt_locations;
    std::set<Eigen::Vector2d,comp_Point<2>> evaluation_locations = gt_locations;
    
    // Annotator
    std::shared_ptr<LandmarkAnnotator<2>> landmark_annotator = nullptr;
    std::shared_ptr<PrecomputedLandmarkAnnotator<2>> simulated_annotator(new PrecomputedLandmarkAnnotator<2>);
    if (human_annotator)
    {
        // Database name
        std::string database_filename = data_folder + "/database-manual-annotations/" + pair_identifier + ".txt";
        landmark_annotator = std::shared_ptr<LandmarkAnnotator<2>>(new HumanLandmarkAnnotator_2D(database_filename,&fixed_image,&moving_image,&partial_window_fixed_image, &partial_window_moving_image,default_radius,confidence_level));
    }
    else
    {
        simulated_annotator->createfromBimodalLognormalDistribution(gt_locations,true_transformation,default_radius,confidence_level,mean_log_eigenvalues+ std::log(scale_user),std_log_eigenvalues + std::log(scale_user),0,rng,pretend_locations_are_certain);
        landmark_annotator = simulated_annotator;
    }
                    

    std::string experiment_identifier = "MyExperiment";


    if ((display) || (human_annotator))
    {
        partial_window_fixed_image.reset(fixed_image);
        partial_window_moving_image.reset(moving_image);
        imshow(partial_window_fixed_image,fixed_image);
        imshow(partial_window_moving_image,moving_image);
        cv::waitKey(1);
    }
    
    // Visualization object
    VisualisationHelper<2> visualisation_helper(fixed_image,moving_image,subsampling_dense_maps,create_dense_maps,spacing,display);


    // Locations of interest for precomputations
    std::set<Eigen::Vector2d,comp_Point<2>> all_locations_of_interest; // this is the set of points that will need to be considered
    union_sets(all_locations_of_interest,salient_locations,evaluation_locations);
    if (visualisation_helper.create_dense_maps())
    {
        std::set<Eigen::Vector2d,comp_Point<2>> image_domain_locations = visualisation_helper.image_domain_for_display();
        add_to_set(all_locations_of_interest,image_domain_locations);
    }

    // Transformations to evaluate
    int nb_elastix_parameters = elastix_parameter_files.size();
    std::vector<std::map<Eigen::Vector2d,Eigen::Vector2d,comp_Point<2>>> transformations_to_evaluate(nb_elastix_parameters);
    for (int ep=0; ep<nb_elastix_parameters; ++ep)
    {
        std::ostringstream oss_ep;
        oss_ep << ep;
        std::string elastix_config_file = "../data/registration-parameter-files/cima/" + elastix_parameter_files[ep];
            
        register_and_transform_with_elastix(transformations_to_evaluate[ep],all_locations_of_interest,fixed_image,moving_image,pair_identifier + "-EP" + oss_ep.str(),stored_registrations_folder,elastix_config_file,load_precomputed_registrations,remove_created_files,coordinateType);
    }
    
    // we define the Gaussian process model (mean and covariance functions)
    std::shared_ptr<Identity<2>> identity(new Identity<2>);
    std::shared_ptr<KernelBundle<2>> multiscale_wendland(new KernelBundle<2>);
    create_multiscale_wendland<2>(*multiscale_wendland,nb_wendland_levels,min_kernel_radius,initial_scale);
    
    // Convert all the aboves in structures
    GaussianProcessPrior<2> gp_prior(identity, multiscale_wendland, training_annotations);
    IOSettings io_settings(results_folder, pair_identifier, experiment_identifier);
    InputTransformations<2> input_transformations(transformations_to_evaluate, true_transformation);
    InputLocations<2> input_locations(salient_locations, target_locations, evaluation_locations);
    Settings settings(annotation_budget);
    
    // The output
    std::vector<Gaussian_Process_Observation<2>> annotated_locations;
    evaluate_query_method<2>(annotated_locations,io_settings,gp_prior,input_locations,settings,input_transformations,rng,querying_strategy,*landmark_annotator,visualisation_helper);
}




void run_cobralab_example()
{
    
    using PixelType = unsigned char;
    using LabelType = unsigned char;
    using ImageType = itk::Image<PixelType,3>;
    using LabelImageType = itk::Image<LabelType,3>;
    
    // Define set of baselines
    QueryingStrategy querying_strategy(entropy,false);
    bool run_on_target_structure = true;
    int evaluation_grid_size = 5; // if run_on_target_structure = false
    LabelType target_label = 2; // if run_on_target_structure = true
    
    // Parameters for generqtion of synthetic deformation (in pixels)
    double std_gaussian = 3;
    double max_svf = 15;
    double reduction_factor = 0.01;
    
    // Random number generator
    boost::random::mt19937 rng;
    
    // Settings
    double isotropic_spacing = 0.6; // in mm
    double default_radius = 1*isotropic_spacing;
    double min_radius = 5*isotropic_spacing;
    double max_radius = 20*isotropic_spacing;
    double confidence_level = 0.01;
    int nb_queries_for_burn_in = 0;
    int nb_training_cases = 1;
    int min_nb_target_samples(10); // to remove small objects
    int max_nb_target_samples(1000); // subsampling for computational reasons
    int annotation_budget(301);
    std::vector<std::string> elastix_parameter_files = {"elastixconfig0.txt","elastixconfig10.txt","elastixconfig15.txt"};
    int nb_brains = 2;
    int ind_testing_brain = 1; // if set to 1, we test on brain1 and use brain0 as a training pair. If set to 0, we do the opposite: test on brain1 and train on brain0.
    bool load_precomputed_registrations = false; // should stay false here
    Eigen::Vector3d spacing(isotropic_spacing,isotropic_spacing,isotropic_spacing);
    CoordinateType coordinate_type = CoordinateType::realworld;
    double initial_scale = 10000;
    int nb_wendland_levels = 6;
    int subsampling_dense_maps = 3;
    int min_radius_in_pixels = 10;
    double min_kernel_radius = std::min<int>(spacing(0)*min_radius_in_pixels,std::min<int>(spacing(1)*min_radius_in_pixels,spacing(2)*min_radius_in_pixels));
    
    std::string folder_stored_registrations = "../data/example-cobralab/precomputed-registrations";
    std::string data_folder = "../data/example-cobralab/cobralab-dataset";
    std::string results_folder = "../results/example-cobralab";
    std::string registration_parameters_folder = "../data/registration-parameter-files/cobralab";
    
    std::vector<ImageType::Pointer> fixed_images(nb_brains), moving_images(nb_brains);
    std::vector<LabelImageType::Pointer> label_images(nb_brains), brain_masks(nb_brains);
    std::vector<std::set<Eigen::Vector3d,comp_Point<3>>> salient_locations(nb_brains);
    std::vector<std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>> true_transformations(nb_brains);
    std::vector<PrecomputedLandmarkAnnotator<3>> landmark_annotators(nb_brains);
        
    // This first loop serves as data preparation: we load the image data, create synthetic transformations to generate the moving image and extract the salient locations in the fixed image
    // This is done for the two available brains
    for (int ind_brain=0; ind_brain<nb_brains; ++ind_brain)
    {
        
        // File names
        std::ostringstream oss_ind_brain;
        oss_ind_brain << ind_brain;
        std::string brain_filename = "brain" + oss_ind_brain.str(); 
        
        // Load images
        std::string t1_image_filename = data_folder + "/brain" + oss_ind_brain.str() + "-t1.nii.gz"; // t1 image 
        std::string t2_image_filename = data_folder + "/brain" + oss_ind_brain.str() + "-t2.nii.gz"; // t2 image 
        std::string labels_filename = data_folder + "/brain" + oss_ind_brain.str() + "-labels.nii.gz"; // segmentation labels  
        ImageType::Pointer t1_image = read_nifti_file<PixelType>(t1_image_filename);
        ImageType::Pointer t2_image_raw = read_nifti_file<PixelType>(t2_image_filename);
        label_images[ind_brain] = read_nifti_file<LabelType>(labels_filename);
        label_images[ind_brain]->SetOrigin(t1_image->GetOrigin());
        label_images[ind_brain]->SetDirection(t1_image->GetDirection());
        label_images[ind_brain]->SetSpacing(t1_image->GetSpacing());

        // We resample T2 to the same dimensions, coordinates size etc as T1
        itk::ResampleImageFilter<ImageType, ImageType>::Pointer resample_filter = itk::ResampleImageFilter<ImageType, ImageType>::New();
        resample_filter->SetReferenceImage(t1_image);
        resample_filter->SetInput(t2_image_raw);
        resample_filter->SetUseReferenceImage(true);
        resample_filter->Update();
        ImageType::Pointer t2_image = resample_filter->GetOutput();

        // Extracts brain mask from labels
        std::cout << "Threshold mask" << std::endl;
        itk::BinaryThresholdImageFilter<LabelImageType,LabelImageType>::Pointer threshold_filter = itk::BinaryThresholdImageFilter<LabelImageType,LabelImageType>::New();
        threshold_filter->SetInput(label_images[ind_brain]);
        threshold_filter->SetLowerThreshold(1);
        threshold_filter->SetUpperThreshold(255);
        threshold_filter->Update();
        brain_masks[ind_brain] = threshold_filter->GetOutput();

        // Strip T1 image
        std::cout << "Strip T1 image" << std::endl;
        itk::MaskImageFilter<ImageType,LabelImageType,ImageType>::Pointer mask_filter = itk::MaskImageFilter<ImageType,LabelImageType,ImageType>::New();
        mask_filter->SetInput1(t1_image);
        mask_filter->SetInput2(brain_masks[ind_brain]);
        mask_filter->Update();
        fixed_images[ind_brain] = mask_filter->GetOutput();  // sets the stripped t1 image as fixed image

        // Extract set of salient points in the fixed image
        std::cout << "Set of salient locations" << std::endl;
        itk::BinaryBallStructuringElement<unsigned char, 3> struct_el;
        struct_el.SetRadius(5); // for the extraction of salient points, we slightly erode the mask further to avoid edge effects
        struct_el.CreateStructuringElement();
        typename itk::BinaryErodeImageFilter<LabelImageType,LabelImageType,itk::BinaryBallStructuringElement<unsigned char, 3>>::Pointer erosion_filter = itk::BinaryErodeImageFilter<LabelImageType,LabelImageType,itk::BinaryBallStructuringElement<unsigned char, 3>>::New();
        erosion_filter->SetForegroundValue(255);
        erosion_filter->SetKernel(struct_el);
        erosion_filter->SetInput(brain_masks[ind_brain]);
        erosion_filter->Update();
        detect_salient_locations<PixelType>(salient_locations[ind_brain],fixed_images[ind_brain],erosion_filter->GetOutput());

        // Create moving image by synthetically deforming the t2 image
        ImageType::RegionType region = fixed_images[ind_brain]->GetLargestPossibleRegion();
        ImageType::SizeType size_fixed_image = region.GetSize(); 
        int dimx = size_fixed_image[0];
        int dimy = size_fixed_image[1];
        int dimz = size_fixed_image[2]; 
        BufferImage3D<Eigen::Vector3d> deformation_map(dimx,dimy,dimz);
        create_synthetic_transformation(moving_images[ind_brain],deformation_map,t2_image,brain_masks[ind_brain],std_gaussian,max_svf,reduction_factor,rng);

        // Load true transformation
        std::cout << "Load true transformation" << std::endl;
        load_true_transformation_from_deformation_map<LabelType>(true_transformations[ind_brain],deformation_map,brain_masks[ind_brain],255);
            
        // Precompute the provided annotations
        std::cout << "Create annotations" << std::endl;
        //std::set<Eigen::Vector3d,comp_Point<3>> gt_locations = create_set_from_map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>(true_transformations[ind_brain]);
        landmark_annotators[ind_brain].createfromBimodalDistribution(salient_locations[ind_brain],true_transformations[ind_brain],default_radius,min_radius,max_radius,confidence_level,0,rng);
            
            
    }
    
    int ind_training_brain;
    if (ind_testing_brain==0)
        ind_training_brain = 1;
    else if (ind_testing_brain==1)
        ind_training_brain = 0;
    else
    {
        std::cout << "ind_testing_brain must be either 0 and 1" << std::endl;
        throw std::invalid_argument("ind_testing_brain must be either 0 and 1");
    }
        

    // Load training annotations
    std::vector<Gaussian_Process_Observation<3>> all_prior_observations = landmark_annotators[ind_training_brain].getAllAnnotations();
    std::random_shuffle(all_prior_observations.begin(),all_prior_observations.end());
    std::vector<std::vector<Gaussian_Process_Observation<3>>> training_annotations(1);
    for (int k=0; k<annotation_budget; ++k)
        training_annotations[0].push_back(all_prior_observations[k]);
   
            
    std::ostringstream oss_ind_testing_brain;
    oss_ind_testing_brain << ind_testing_brain;
    std::string pair_identifier = "brain" + oss_ind_testing_brain.str();
    
    std::set<Eigen::Vector3d,comp_Point<3>> target_locations, evaluation_locations;
    std::string experiment_identifier;
    if (run_on_target_structure)
    {
        LabelImageType::Pointer edge_map_object = create_mask_edge_object(label_images[ind_testing_brain],target_label);
                        
        // Multiply the edge mask by the brain mask
        itk::MaskImageFilter<LabelImageType,LabelImageType,ImageType>::Pointer target_mask_filter = itk::MaskImageFilter<LabelImageType,LabelImageType,ImageType>::New();
        target_mask_filter->SetInput1(edge_map_object);
        target_mask_filter->SetInput2(brain_masks[ind_testing_brain]);
        target_mask_filter->Update();
        edge_map_object = target_mask_filter->GetOutput();     

        std::set<Eigen::Vector3d,comp_Point<3>> edge_points;
        load_locations_of_interest_from_mask<LabelType>(edge_points,edge_map_object,255);
        create_random_set<Eigen::Vector3d,comp_Point<3>>(target_locations,edge_points,std::min<int>(edge_points.size(),max_nb_target_samples));

        std::ostringstream oss_t;
        oss_t << (int)target_label;
        experiment_identifier = "Target" + oss_t.str();
                        
        if (target_locations.size()<min_nb_target_samples)
            throw std::runtime_error("Not enough target points within the specifed structure");
       
        evaluation_locations = target_locations;
    }
    else
    {

        experiment_identifier = "FullBrain";
        target_locations = salient_locations[ind_testing_brain]; // equivalent to set full mask
        load_locations_of_interest_from_mask<LabelType>(evaluation_locations,brain_masks[ind_testing_brain],255,evaluation_grid_size);
    }
    
    
    // All locations of interest for precomputations
    std::set<Eigen::Vector3d,comp_Point<3>> all_locations_of_interest;
    union_sets(all_locations_of_interest,salient_locations[ind_testing_brain],evaluation_locations);
    
    
    // Transformations to evaluate
    int nb_elastix_parameters = elastix_parameter_files.size();
    std::vector<std::map<Eigen::Vector3d,Eigen::Vector3d,comp_Point<3>>> transformations_to_evaluate(nb_elastix_parameters);
    for (int ep=0; ep<nb_elastix_parameters; ++ep)
    {
        bool delete_registration_files = true;
            
        std::ostringstream oss_ep;
        oss_ep << ep;
        std::string elastix_config_file = registration_parameters_folder + "/" + elastix_parameter_files[ep];
            
        register_and_transform_with_elastix<PixelType,LabelType>(transformations_to_evaluate[ep],all_locations_of_interest,fixed_images[ind_testing_brain],moving_images[ind_testing_brain],pair_identifier + "-EP" + oss_ep.str(),folder_stored_registrations,elastix_config_file,load_precomputed_registrations,delete_registration_files,coordinate_type,brain_masks[ind_testing_brain]);
    }
            
            
    VisualisationHelper<3> visualisation_helper(spacing); // void - not implemented in 3d        
    
     // We define the Gaussian process model (mean and covariance functions)
    std::shared_ptr<Identity<3>> identity(new Identity<3>);
    std::shared_ptr<KernelBundle<3>> multiscale_wendland(new KernelBundle<3>);
    create_multiscale_wendland<3>(*multiscale_wendland,nb_wendland_levels,min_kernel_radius,initial_scale);
    
    // Convert all the aboves in structures
    GaussianProcessPrior<3> gp_prior(identity, multiscale_wendland, training_annotations);
    IOSettings io_settings(results_folder, pair_identifier, experiment_identifier);
    InputTransformations<3> input_transformations(transformations_to_evaluate, true_transformations[ind_testing_brain]);
    InputLocations<3> input_locations(salient_locations[ind_testing_brain], target_locations, evaluation_locations);
    Settings settings(annotation_budget);
    
    // The output
    std::vector<Gaussian_Process_Observation<3>> annotated_locations;
    evaluate_query_method<3>(annotated_locations,io_settings,gp_prior,input_locations,settings,input_transformations,rng,querying_strategy,landmark_annotators[ind_testing_brain],visualisation_helper);    
}

int main()
{
    run_cima_example();
    run_cobralab_example();
}


