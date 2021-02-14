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


