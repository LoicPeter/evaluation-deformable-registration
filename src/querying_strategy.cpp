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


#include "querying_strategy.hpp"


QueryingStrategy::QueryingStrategy()
{}

QueryingStrategy::QueryingStrategy(bool pick_at_random, bool reestimate_variances, bool entropy_only, bool dense_target)
{
  //  m_pick_at_random = pick_at_random;
    if (entropy_only)
        m_uncertainty_measure = entropy;
    else
    {
        if (pick_at_random)
            m_uncertainty_measure = random_placement;
    }
    m_reestimate_variances = reestimate_variances;
 //   m_entropy_only = entropy;
    m_dense_target = dense_target;
}

QueryingStrategy::QueryingStrategy(UncertaintyMeasure uncertainty_measure, bool reestimate_variances, bool dense_target)
{
    m_uncertainty_measure = uncertainty_measure;
    m_reestimate_variances = reestimate_variances;
    m_dense_target = dense_target;
}

std::string QueryingStrategy::get_str() const
{
    std::string res = "Unnamed";
    
    if (m_uncertainty_measure == random_placement)
        res = "Random";
    if (m_uncertainty_measure == entropy)
        res = "Entropy";
    if (m_uncertainty_measure == heuristic)
        res = "Heuristic";
    if (m_reestimate_variances)
        res = res + "-UpdateVariances";
    if (m_dense_target)
        res = res + "-DenseTarget";
    return res;    
}


