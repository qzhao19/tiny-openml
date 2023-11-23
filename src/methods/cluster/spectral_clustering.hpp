#ifndef METHODS_CLUSTER_SPECTRAL_CLUSTERING_HPP
#define METHODS_CLUSTER_SPECTRAL_CLUSTERING_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace cluster {

/**
 * K-Means clustering.
*/
template<typename DataType>
class SpectralClustering {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVecType = Eigen::Matrix<DataType, 1, Eigen::Dynamic>;
    using ColVecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    using IdxMatType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;
    
    std::string init_;
    std::size_t num_init_;
    std::size_t num_clusters_;
    std::size_t max_iter_;
    // double tol;
    double gamma_;
    
    MatType centroids_;
    MatType laplace_m_;
    MatType eig_vals_;
    MatType eig_vecs_;

protected:
    void laplace_matrix(const MatType& X) {
        std::size_t num_rows = X.rows(), num_cols = X.cols();
        // build the matrix
        MatType W(num_rows, num_rows);
        for (std::size_t i = 0; i < num_rows; ++i) {
            for (std::size_t j = i; j < num_rows; ++j) {
                DataType w = std::exp((X.col(i) - X.cols(j)).array().pow(2.0).sum() / (-2.0 * gamma_ * gamma_));
                W(i, j) = w;
                W(j, i) = w;
            }
        }
        MatType D;
        D = W.colwise().sum().asDiagonal();
        laplace_m_ = D - W;
    }

public:
    SpectralClustering(): init_("kmeans++"), 
        num_init_(10),
        num_clusters_(3), 
        max_iter_(300), 
        gamma_(1.0) {};

    SpectralClustering(std::string init,
        std::size_t num_init,
        std::size_t num_clusters,
        std::size_t max_iter, 
        double tol): init_(init), 
            num_init_(num_init),
            num_clusters_(num_clusters), 
            max_iter_(max_iter), 
            gamma_(tol) {};

    ~SpectralClustering() {};
    /**
     * fit dataset to compute k-means clustering
    */
    void fit_transform(const MatType& X) {
        std::tie(eig_vals_, eig_vecs_) = math::eig<MatType>(laplace_m_);
    }

    /**
     * Predict the labels for the input dataset
     * @param X  ndarray of shape [num_samples, num_features]
     *    the input dataset
    */
    const VecType predict(const MatType& X) {
        return ;
    }   

};

} // cluster_model
} // openml

#endif /*METHODS_CLUSTER_SPECTRAL_CLUSTERING_HPP*/
