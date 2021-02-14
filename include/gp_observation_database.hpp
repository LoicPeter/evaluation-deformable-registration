#ifndef GP_OBSERVATION_DATABASE_H_INCLUDED
#define GP_OBSERVATION_DATABASE_H_INCLUDED

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <map>

template <int d>
class comp_Point 
{
    public:
    bool operator()(const Eigen::Matrix<double,d,1>& p_1, const Eigen::Matrix<double,d,1>& p_2) const;
};

template <int d>
bool comp_Point<d>::operator()(const Eigen::Matrix<double,d,1>& p_1, const Eigen::Matrix<double,d,1>& p_2) const
{
    for (int k=(d-1); k>=0; --k)
    {
        if (p_1(k)<p_2(k))
            return true;
        else
        {
            if (p_1(k)>p_2(k))
                return false;
        }
    }

    return false;
}


template <int d>
struct Gaussian_Process_Observation
{
    Eigen::Matrix<double,d,1> input_point;
    Eigen::Matrix<double,d,1> output_point;
    Eigen::Matrix<double,d,d> observation_noise_covariance_matrix;
};

template <int d>
struct Noisy_Point
{
    Eigen::Matrix<double,d,1> mean; 
    Eigen::Matrix<double,d,d> covariance_matrix;    
};

template <int d>
class GP_Observation_Database
{
    public:
    void save(const std::string& filename) const;
    void add_to_database(const Gaussian_Process_Observation<d>& obs);
    void load(const std::string& filename);
    bool is_obs_available(const Eigen::Matrix<double,d,1>& pt) const {return (m_observations_map.count(pt));};
    void retrieve_observation(Eigen::Matrix<double,d,1>& output_point, Eigen::Matrix<double,d,d>& observation_noise_covariance_matrix, const Eigen::Matrix<double,d,1>& input_point) const;
    
    private:
    std::map<Eigen::Matrix<double,d,1>,Noisy_Point<d>,comp_Point<d>> m_observations_map;
};


template <int d>
void GP_Observation_Database<d>::save(const std::string& filename) const
{
    std::ofstream infile(filename);
    typename std::map<Eigen::Matrix<double,d,1>,Noisy_Point<d>,comp_Point<d>>::const_iterator it;
    for (it=m_observations_map.begin(); it!=m_observations_map.end(); ++it)
    {
        Eigen::Matrix<double,d,1> input_point = it->first;
        Noisy_Point<d> output_point = it->second;
        Eigen::Matrix<double,d,d> cov = output_point.covariance_matrix;
        for (int k=0; k<d; ++k)
            infile << input_point(k) << " ";
        for (int k=0; k<d; ++k)
            infile << output_point.mean(k) << " ";
        for (int i=0; i<d; ++i)
        {
            for (int j=0; j<d; ++j)
            {
                infile << cov(i,j) << " ";
            }
        }
        infile << std::endl;
    }
    infile.close();
}

template <int d>
void GP_Observation_Database<d>::add_to_database(const Gaussian_Process_Observation<d>& obs)
{
    Noisy_Point<d> noisy_output_point;
    noisy_output_point.mean = obs.output_point;
    noisy_output_point.covariance_matrix = obs.observation_noise_covariance_matrix;
    m_observations_map[obs.input_point] = noisy_output_point;
}

template <int d>
void GP_Observation_Database<d>::load(const std::string& filename)
{
    m_observations_map.clear();
    int count_points_loaded(0);
    std::ifstream infile(filename);
    if (infile.good())
    {
        while (!infile.eof())
        {
            Gaussian_Process_Observation<d> obs;
            
            for (int k=0; k<d; ++k)
                infile >> obs.input_point(k);
            for (int k=0; k<d; ++k)
                infile >> obs.output_point(k);
            for (int i=0; i<d; ++i)
            {
                for (int j=0; j<d; ++j)
                    infile >> obs.observation_noise_covariance_matrix(i,j);
            }
            
            this->add_to_database(obs);
            ++count_points_loaded;
        }
    }
    std::cout << count_points_loaded << " points loaded" << std::endl;
}

template <int d>
void GP_Observation_Database<d>::retrieve_observation(Eigen::Matrix<double,d,1>& output_point, Eigen::Matrix<double,d,d>& observation_noise_covariance_matrix, const Eigen::Matrix<double,d,1>& input_point) const
{
    Noisy_Point<d> noisy_output_point = m_observations_map.at(input_point);
    output_point = noisy_output_point.mean;
    observation_noise_covariance_matrix = noisy_output_point.covariance_matrix;
}


#endif
