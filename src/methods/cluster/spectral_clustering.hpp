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

protected:

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
    void fit(const MatType& X) {
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
