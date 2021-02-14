#ifndef MEAN_FUNCTIONS_INCLUDED
#define MEAN_FUNCTIONS_INCLUDED

#include <vector>
#include <Eigen/Dense>
#include "gp_observation_database.hpp"

template <int d>
class MeanFunction
{
    public:
    virtual ~MeanFunction() = 0;
    virtual MeanFunction<d>* new_clone() const = 0;
    virtual Eigen::Matrix<double,d,1> operator()(const Eigen::Matrix<double,d,1>& x) const = 0;
    virtual void fit(const std::vector<Gaussian_Process_Observation<d>>& known_values) = 0;
};

template <int d>
class Identity : public MeanFunction<d>
{
    public:
    ~Identity();
    MeanFunction<d>* new_clone() const;
    Eigen::Matrix<double,d,1> operator()(const Eigen::Matrix<double,d,1>& x) const;
    void fit(const std::vector<Gaussian_Process_Observation<d>>& known_values);
};

template <int d>
class Translation : public MeanFunction<d>
{
    public:
    Translation(); 
    Translation(const Eigen::Matrix<double,d,1>& displacement); 
    ~Translation();
    void set_displacement(const Eigen::Matrix<double,d,1>& displacement);
    MeanFunction<d>* new_clone() const;
    Eigen::Matrix<double,d,1> operator()(const Eigen::Matrix<double,d,1>& x) const;
    void fit(const std::vector<Gaussian_Process_Observation<d>>& known_values);
    
    private:
    Eigen::Matrix<double,d,1> m_displacement;
};


template <int d>
MeanFunction<d>::~MeanFunction()
{}

template <int d>
Identity<d>::~Identity()
{}

template <int d>
void Identity<d>::fit(const std::vector<Gaussian_Process_Observation<d>>& known_values)
{
    std::cout << "Try to fit identity, no changes made" << std::endl;
}

template <int d>
MeanFunction<d>* Identity<d>::new_clone() const
{
    return new Identity<d>();
}

template <int d>
Eigen::Matrix<double,d,1> Identity<d>::operator()(const Eigen::Matrix<double,d,1>& p) const
{
    return p;
}

template <int d>
Translation<d>::Translation()
{
    this->m_displacement = Eigen::Matrix<double,d,1>::Zero();
}

template <int d>
Translation<d>::Translation(const Eigen::Matrix<double,d,1>& displacement)
{
    this->m_displacement = displacement;
}

template <int d>
Translation<d>::~Translation()
{}

template <int d>
void Translation<d>::set_displacement(const Eigen::Matrix<double,d,1>& displacement)
{
    this->m_displacement = displacement;
}

template <int d>
void Translation<d>::fit(const std::vector<Gaussian_Process_Observation<d>>& known_values)
{
    Eigen::Matrix<double,d,1> average_displacement = Eigen::Matrix<double,d,1>::Zero();
    int nb_known_values = known_values.size();
    for (int ind=0; ind<nb_known_values; ++ind)
        average_displacement = average_displacement + known_values[ind].output_point - known_values[ind].input_point;
    average_displacement = (1/((double)nb_known_values))*average_displacement;
    this->set_displacement(average_displacement);
}

template <int d>
MeanFunction<d>* Translation<d>::new_clone() const
{
    return new Translation<d>(this->m_displacement);
}

template <int d>
Eigen::Matrix<double,d,1> Translation<d>::operator()(const Eigen::Matrix<double,d,1>& p) const
{
    return (p + this->m_displacement);
}





#endif // REGISTRATION_AVERAGING_INCLUDED
