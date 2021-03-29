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


#include "updatable_cholesky.hpp"
#include <Eigen/LU>
#include <iostream>


// In-house Cholesky decomposition supporting updates

UpdatableCholesky::UpdatableCholesky()
{
    Eigen::LLT<Eigen::MatrixXd>();
}

UpdatableCholesky::UpdatableCholesky(const Eigen::MatrixXd& P)
{
    this->compute(P);
}

// We update the matrix P by adding a column Q on the right and a square block S
void UpdatableCholesky::update(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S)
{
    Eigen::MatrixXd L = this->matrixL();
    Eigen::MatrixXd P = L*(L.transpose()); // this step recovers the matrix P of which L is the Cholesky factor
    this->update(Q,S,P);
}


// Much faster if we provide a matrix P which can be either the matrix before or after update (slightly faster if we provide the latter)
void UpdatableCholesky::update(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S, const Eigen::MatrixXd& P)
{
   //  std::cout << "Update Cholesky" << std::endl;
  //  Eigen::MatrixXd R = Q.transpose();
    int d = S.rows();
    int current_size = m_matrix.rows();
    int new_size = current_size + d;
    
    // Compute updated factors
    Eigen::MatrixXd L = this->matrixL();
    Eigen::PartialPivLU<Eigen::MatrixXd> L_trivial_LU = L.lu();
    
    Eigen::MatrixXd R2 = (L_trivial_LU.solve(Q)).transpose();
    Eigen::LLT<Eigen::MatrixXd> S2_chol = (S - R2*(R2.transpose())).llt();
    Eigen::MatrixXd S2 = S2_chol.matrixL();
    
    // Update Cholesky decomposition
    m_matrix.conservativeResize(new_size,new_size);
    m_matrix.block(current_size,0,d,current_size) = R2;
    m_matrix.block(current_size,current_size,d,d) = S2;
    
    // Update the L1 norm
    Eigen::MatrixXd P_updated;
    if ((P.rows()==new_size) && (P.cols()==new_size)) // if the provided matrix is the matrix after update
        m_l1_norm = P.colwise().lpNorm<1>().maxCoeff();
    else
    {
        if ((P.rows()==current_size) && (P.cols()==current_size)) // if it is the matrix before update, we update it first
        {
            Eigen::MatrixXd P_updated = P;
            P_updated.conservativeResize(new_size,new_size);
            P_updated.block(0,current_size,current_size,d) = Q;
            P_updated.block(current_size,0,d,current_size) = Q.transpose();
            P_updated.block(current_size,current_size,d,d) = S; 
            m_l1_norm = P_updated.colwise().lpNorm<1>().maxCoeff();
        }
        else
            std::cout << "In the Cholesky update; the size of the provided matrix P does not match" << std::endl;
    }
    
 //   std::cout << "Done" << std::endl;
    
    // Sanity check
//     Eigen::MatrixXd updated_L = this->matrixL();
//     Eigen::MatrixXd recomputed_L;
//     if ((P.rows()==new_size) && (P.cols()==new_size))
//         P_updated = P;
//     recomputed_L = P_updated.llt().matrixL();
//     Eigen::MatrixXd difference = updated_L - recomputed_L;
//     double max_error = difference.lpNorm<Eigen::Infinity>();
//     std::cout << "Accuracy of Cholesky update: " << max_error << std::endl;
 //  std::cout << "Updated L:" << std::endl << updated_L << std::endl;
 //  std::cout << "Recomputed L:" << std::endl << recomputed_L << std::endl;
}


// This updates the inverse given blockwise by (P, Q; R, S), for which we already know (and only need to know) the inverse of P
void efficient_update_matrix_inverse(Eigen::MatrixXd& updated_inverse, const Eigen::MatrixXd& P_inv, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& S)
{
    int size_P = P_inv.rows(); // number of rows of the (big) matrix P
    int d = R.rows(); // number of rows added
    
    
    Eigen::MatrixXd M = (S - R*P_inv*Q).inverse(); // square but not symmetric in general
            
    // Update inverse
    updated_inverse.resize(size_P + d,size_P + d);
    updated_inverse.block(size_P,size_P,d,d) = M;
    Eigen::MatrixXd R2 = -M*R*P_inv;
    Eigen::MatrixXd Q2 = -P_inv*Q*M;
    updated_inverse.block(size_P,0,d,size_P) = R2;
    updated_inverse.block(0,size_P,size_P,d) = Q2;
    updated_inverse.block(0,0,size_P,size_P) = P_inv - P_inv*Q*R2; // faster    
}

// Assuming a matrix P' = (P, Q; R, S) and B' = (B ; new_row_B), 
// If we know already the product (P_inv*B), this function outputs efficiently the new product (P'_inv*B')
// We further assume here that P'inv has been computed (updated_inverse)
void efficient_update_matrix_inverse_times_B(Eigen::MatrixXd& updated_product, const Eigen::MatrixXd& current_product, const Eigen::MatrixXd& new_row_B, const Eigen::MatrixXd& R, const Eigen::MatrixXd& updated_inverse)
{
    int nb_cols_B = current_product.cols();
    int nb_rows_updated_inverse = updated_inverse.rows();
    int d = new_row_B.rows();
    int size_P = nb_rows_updated_inverse - d;
    updated_product.resize(nb_rows_updated_inverse,nb_cols_B);
    
    Eigen::MatrixXd Q2 = updated_inverse.block(0,size_P,size_P,d);
    Eigen::MatrixXd M = updated_inverse.block(size_P,size_P,d,d);
          //  precomputed_K_tilde_XX_inv_times_K_XS.block(0,0,ind*d,size_S*d) = current_K_tilde_XX_inv_times_K_XS + P_inv*Q*M*R*current_K_tilde_XX_inv_times_K_XS + (R2.transpose())*new_row_K_XS;
    
    Eigen::MatrixXd temp = R*current_product;
    updated_product.block(0,0,size_P,nb_cols_B) = current_product - Q2*(temp - new_row_B); // faster
    updated_product.block(size_P,0,d,nb_cols_B) = M*(new_row_B - temp);
}

void efficient_update_matrix_inverse_times_B(Eigen::MatrixXd& updated_product, const Eigen::MatrixXd& current_product, const Eigen::MatrixXd& new_row_B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S, const Eigen::LLT<Eigen::MatrixXd>& P_cholesky)
{
    int nb_cols_B = current_product.cols();
    int nb_rows_P = Q.rows();
    int d = new_row_B.rows();
    updated_product.resize(nb_rows_P + d,nb_cols_B);
    Eigen::MatrixXd R = Q.transpose();
    
    Eigen::MatrixXd P_inv_Q = P_cholesky.solve(Q);
    Eigen::MatrixXd M = (S - R*P_inv_Q).inverse();
    Eigen::MatrixXd P_inv_Q_M = P_inv_Q*M;
    Eigen::MatrixXd R_cp = R*current_product;

    updated_product.block(0,0,nb_rows_P,nb_cols_B) = current_product + P_inv_Q_M*(R_cp - new_row_B); // faster
    updated_product.block(nb_rows_P,0,d,nb_cols_B) = M*(new_row_B - R_cp);
}



