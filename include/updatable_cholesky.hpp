#ifndef UPDATABLE_CHOLESKY_H_INCLUDED
#define UPDATABLE_CHOLESKY_H_INCLUDED

#include <Eigen/Cholesky>

class UpdatableCholesky : public Eigen::LLT<Eigen::MatrixXd>
{
    public:
    UpdatableCholesky();
    UpdatableCholesky(const Eigen::MatrixXd& P);
    void update(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S);
    void update(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S, const Eigen::MatrixXd& P);
};



void efficient_update_matrix_inverse(Eigen::MatrixXd& updated_inverse, const Eigen::MatrixXd& P_inv, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& S);

void efficient_update_matrix_inverse_times_B(Eigen::MatrixXd& updated_product, const Eigen::MatrixXd& current_product, const Eigen::MatrixXd& new_row_B, const Eigen::MatrixXd& R, const Eigen::MatrixXd& updated_inverse);

void efficient_update_matrix_inverse_times_B(Eigen::MatrixXd& updated_product, const Eigen::MatrixXd& current_product, const Eigen::MatrixXd& new_row_B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S, const Eigen::LLT<Eigen::MatrixXd>& P_cholesky);


#endif
