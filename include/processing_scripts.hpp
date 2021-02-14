#ifndef PROCESSING_SCRIPTS_INCLUDED
#define PROCESSING_SCRIPTS_INCLUDED

#include <set>
#include <map>
#include <string>
#include <vector>
#include "gp_observation_database.hpp"
#include "querying_strategy.hpp"
#include <Eigen/Core>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

enum class CoordinateType {index, realworld};


template <typename T, typename comp_T>
void create_random_sets(std::vector<std::set<T,comp_T>>& output_sets, const std::set<T,comp_T>& input_full_set, const std::vector<int>& size_sets)
{
    int nb_elements = input_full_set.size();
    int nb_sets = size_sets.size();
    output_sets.clear();
    output_sets.resize(nb_sets);

    std::vector<int> list_indices(nb_elements);
    for (int n=0; n<nb_elements; ++n)
        list_indices[n] = n;
    
    for (int ind_set=0; ind_set<nb_sets; ++ind_set)
    {
    
        if (size_sets[ind_set]>nb_elements)
            std::cout << "In create_random_sets: the queried size is larger than the number of elements in the set to draw from" << std::endl;
        else
        {
            std::random_shuffle(list_indices.begin(),list_indices.end());
            std::sort(list_indices.begin(),list_indices.begin()+size_sets[ind_set]);
            
            int i(0);
            int current_index(0);
            typename std::set<T,comp_T>::const_iterator it = input_full_set.begin();
            while (current_index<size_sets[ind_set])
            {
                if (i == list_indices[current_index])
                {
                    output_sets[ind_set].insert(*it);
                    ++current_index;
                }
                
                ++i;
                ++it;
            }
        }
    }

}

// a bit suboptimal since it forces unnecessary copies
template <typename T, typename comp_T>
void create_random_set(std::set<T,comp_T>& output_set, const std::set<T,comp_T>& input_full_set, int size_set)
{
    std::vector<int> size_sets(1,size_set);
    std::vector<std::set<T,comp_T>> output_sets(1);
    create_random_sets<T,comp_T>(output_sets,input_full_set,size_sets);
    output_set = output_sets[0]; 
}


template <typename type_input, typename type_output, typename type_comparison>
std::set<type_input,type_comparison> create_set_from_map(const std::map<type_input,type_output,type_comparison>& input_map)
{
    std::set<type_input,type_comparison> output_set;
    typename std::map<type_input,type_output,type_comparison>::const_iterator it_map;
    for (it_map = input_map.begin(); it_map != input_map.end(); ++it_map)
        output_set.insert(it_map->first);
    return output_set;
}


template <typename T, typename comp_T>
void split_set_K_fold(std::vector<std::set<T,comp_T>>& training_sets, std::vector<std::set<T,comp_T>>& test_sets, const std::set<T,comp_T>& input_full_set, int K)
{
    int nb_elements = input_full_set.size();
    training_sets.clear();
    test_sets.clear();
    training_sets.resize(K);
    test_sets.resize(K);

    std::vector<int> list_indices(nb_elements);
    for (int n=0; n<nb_elements; ++n)
        list_indices[n] = n;
    std::random_shuffle(list_indices.begin(),list_indices.end());

    int i(0);
    for (auto it=input_full_set.begin(); it!=input_full_set.end(); ++it)
    {
        int ind_test_set = list_indices[i] % K;
        for (int k=0; k<K; ++k)
        {
            if (k==ind_test_set)
                test_sets[k].insert(*it);
            else
                training_sets[k].insert(*it);
        }

        ++i;
    }

}

template <int d>
void write_transformix_input_points(const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_points, const std::string& input_points_filename, CoordinateType coordinate_type)
{
    int nb_points = (int)target_points.size();
    std::ofstream infile(input_points_filename);
    if (coordinate_type==CoordinateType::realworld)
        infile << "point" << std::endl;
    else
        infile << "index" << std::endl;
    infile << nb_points << std::endl;
    for (auto it=target_points.begin(); it!=target_points.end(); ++it)
    {
        for (int k=0; k<d; ++k)
            infile << (*it)(k) << " ";
        infile << std::endl;
    }
    infile.close();
}

template <int d>
void read_transformix_output_points(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& transformation, const std::string& output_points_filename, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_set)
{
    std::string line;
    std::ifstream infile(output_points_filename);
    int l(0);
    std::string temp_str;
    Eigen::Matrix<double,d,1> input_point, output_point, input_point_in_target;
    while (std::getline(infile,line,';'))
    {
        
        std::stringstream s_stream(line);
        if ((l % 5)==2)
        {
            for (int k=0; k<3; ++k)
                s_stream >> temp_str;
            for (int k=0; k<d; ++k)
                s_stream >> input_point(k);
            
            // We make sure the point is in the target set for numerical issues
            Eigen::Matrix<double,d,1> input_point_in_target;
            double dist = get_distance_to_set(input_point_in_target,input_point,target_set);
            if (dist<0.001)
                input_point = input_point_in_target;
            else
                
                std::cout << "In read_transformix_output_points, an input point that was read was not found in the target set: dist = " << dist << std::endl;

            
        }
        if ((l % 5)==4)
        {
            for (int k=0; k<3; ++k)
                s_stream >> temp_str;
            for (int k=0; k<d; ++k)
                s_stream >> output_point(k);
            transformation.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(input_point,output_point));
            
//             std::cout << "Input point: ( " ;
//             for (int k=0; k<d; ++k)
//                 std::cout << input_point(k) << " ";
//             std::cout << ") ; Output point: ( ";
//             for (int k=0; k<d; ++k)
//                 std::cout << output_point(k) << " ";
//             std::cout << " )" << std::endl;
        }

     //   std::cout << line << std::endl;
        ++l;
    }
    
    infile.close();
}





// This function takes as input fixed and moving images, the elastix parameters of the registration, and aset of target points in the fixed image that we want to match. It returns a transformation map containing the predicted transformation
template <int d>
void register_and_transform_with_elastix(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_points, const std::string& cmd_register, const std::string& temp_folder, bool load_precomputed_file, CoordinateType coordinate_type)
{
    static_assert((d==2) || (d==3));
    std::string input_points_str = temp_folder + "/elastix_input_points.txt";
    std::string devnull = " >/dev/null";

    std::string cmd_transform = "transformix -def " + input_points_str + " -out " + temp_folder + " -tp " + temp_folder + "/TransformParameters.0.txt -threads 6" + devnull;
    
    std::cout << "Start registration with Elastix" << std::endl;
    
    // Create file with input points
    std::cout << "Write files with input points..." << std::endl;
    write_transformix_input_points(target_points,input_points_str,coordinate_type);
    
    // Register and transform
    int res_command;
    if (!load_precomputed_file)
    {
        std::cout << "Register..." << std::endl;
        res_command = system(cmd_register.c_str()); // create TransformParameters.0.txt in temp folder
    }
    
    std::cout << "Apply transform..." << std::endl;
    res_command = system(cmd_transform.c_str()); // apply the transformation in TransformParameters.0.txt to the set of input points - this creates a file outputpoints.txt in temp_folder
    
    // Read the files with the output points
    std::cout << "Write files with output points..." << std::endl;
    read_transformix_output_points(transformation,temp_folder + "/outputpoints.txt",target_points);
    
    
    std::cout << "Registration with Elastix done" << std::endl;
    
}


// This function takes as input fixed and moving images, the elastix parameters of the registration, and aset of target points in the fixed image that we want to match. It returns a transformation map containing the predicted transformation
template <int d>
void register_and_transform_with_elastix(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_points, const std::string& fixed_image_filename, const std::string& moving_image_filename, const std::string& temp_folder, const std::string& elastix_registration_parameters, bool load_precomputed_file, CoordinateType coordinate_type)
{
    std::string cmd_register = "elastix -f " + fixed_image_filename + " -m " + moving_image_filename + " -out " + temp_folder + " -p " +  elastix_registration_parameters + " -threads 6 >/dev/null";
    register_and_transform_with_elastix<d>(transformation,target_points,cmd_register,temp_folder,load_precomputed_file,coordinate_type);
}

// This function takes as input fixed and moving images, the elastix parameters of the registration, and aset of target points in the fixed image that we want to match. It returns a transformation map containing the predicted transformation
template <int d>
void register_and_transform_with_elastix(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& transformation, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_points, const std::string& fixed_image_filename, const std::string& moving_image_filename, const std::string& mask_in_fixed_filename, const std::string& temp_folder, const std::string& elastix_registration_parameters, bool load_precomputed_file, CoordinateType coordinate_type)
{
    std::string cmd_register = "elastix -f " + fixed_image_filename + " -m " + moving_image_filename + " -fMask " + mask_in_fixed_filename + " -out " + temp_folder + " -p " +  elastix_registration_parameters + " -threads 6 >/dev/null";
    register_and_transform_with_elastix<d>(transformation,target_points,cmd_register,temp_folder,load_precomputed_file,coordinate_type);
}


template <int d>
void add_random_noise_transformation(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& output_transformation, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& true_transformation, double standard_deviation, boost::random::mt19937& rng)
{
    output_transformation.clear();
    boost::normal_distribution<double> normal_dist(0.0,standard_deviation);
    for (auto it_map=true_transformation.begin(); it_map!=true_transformation.end(); ++it_map)
    {
        Eigen::Matrix<double,d,1> added_noise;
        for (int k=0; k<d; ++k)
            added_noise(k) = normal_dist(rng);
        output_transformation[it_map->first] = it_map->second + added_noise;
    }
}

template <int d>
Eigen::Matrix<double,d,1> load_spacing_from_txt_file(const std::string& filename)
{
    Eigen::Matrix<double,d,1> spacing;
    std::ifstream infile(filename);
    for (int k=0; k<d; ++k)
        infile >> spacing(k);
    infile.close();
    return spacing;
}

template <int d>
Eigen::Matrix<double,d,1> apply_spacing(const Eigen::Matrix<double,d,1>& pt, const Eigen::Matrix<double,d,1>& spacing)
{
    Eigen::Matrix<double,d,1> output_pt;
    for (int k=0; k<d; ++k)
        output_pt(k) = pt(k) * spacing(k);
    return output_pt;
}

template <int d>
void apply_spacing(std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& output_set, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& input_set, const Eigen::Matrix<double,d,1>& spacing)
{
    output_set.clear();
    for (const auto& input_pt : input_set)
    {
        Eigen::Matrix<double,d,1> pt = apply_spacing<d>(input_pt,spacing);
        output_set.insert(pt);
    }
}

template <int d>
void apply_spacing(std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& output_map, const std::map<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>,comp_Point<d>>& input_map, const Eigen::Matrix<double,d,1>& spacing)
{
    output_map.clear();
    for (const auto& input_pair : input_map)
    {
        Eigen::Matrix<double,d,1> pt1 = apply_spacing<d>(input_pair.first,spacing);
        Eigen::Matrix<double,d,1> pt2 = apply_spacing<d>(input_pair.second,spacing);
        output_map.insert(std::pair<Eigen::Matrix<double,d,1>,Eigen::Matrix<double,d,1>>(pt1,pt2));
    }
}


void write_elastix_settings_textfiles(const std::string& folder);

void write_elastix_settings_textfiles_cobralab(const std::string& folder);

#endif // REGISTRATION_AVERAGING_INCLUDED
