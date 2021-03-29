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


#ifndef QUERYING_STRATEGY_INCLUDED
#define QUERYING_STRATEGY_INCLUDED

#include <string>
#include <Eigen/Core>
#include <set>
#include "gp_observation_database.hpp"

enum UncertaintyMeasure {random_placement, entropy, heuristic};

class QueryingStrategy
{
    public:
    QueryingStrategy();
    QueryingStrategy(bool pick_at_random, bool reestimate_variances, bool entropy, bool dense_target = false);
    QueryingStrategy(UncertaintyMeasure uncertainty_measure, bool reestimate_variances, bool dense_target = false);
    std::string get_str() const;
    bool m_reestimate_variances;
//    bool m_pick_at_random;
//    bool m_entropy_only;
    bool m_dense_target;
    UncertaintyMeasure m_uncertainty_measure;
  
};


template <int d>
double get_distance_to_set(Eigen::Matrix<double,d,1>& closest_point, const Eigen::Matrix<double,d,1>& input_point, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_set)
{
    double lowest_dist(-1);
    for (const auto& p : target_set)
    {
        Eigen::Matrix<double,d,1> diff = p - input_point;
        double current_dist = diff.norm();
        if ((lowest_dist==(-1)) || current_dist<lowest_dist)
        {
            lowest_dist = current_dist;
            closest_point = p;
        }
    }
        
    return lowest_dist;
}


template <int d>
double get_distance_to_set(const Eigen::Matrix<double,d,1>& input_point, const std::set<Eigen::Matrix<double,d,1>,comp_Point<d>>& target_set)
{
    Eigen::Matrix<double,d,1> dummy_p;
    return get_distance_to_set<d>(dummy_p,input_point,target_set);
}

#endif // REGISTRATION_AVERAGING_INCLUDED
