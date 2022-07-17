#ifndef METHODS_CLUSTER_KMEANS_HPP
#define METHODS_CLUSTER_KMEANS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace cluster {

/**
 * K-Means clustering.
*/
template<typename DataType>
class KMeans {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
    std::string init_;
    std::size_t num_init_;
    std::size_t num_clusters_;
    std::size_t max_iter_;


protected:
    MatType init_centroid(const MatType& X, 
        const VecType& x_squared_norms) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols(); 
        MatType centroids(num_clusters_, num_features);

        if (init_ == "random") {
            IdxType index = math::permutation<IdxType>(num_samples);
            IdxType selected_index = index.topRows(num_clusters_);
            centroids = X(selected_index, Eigen::all);
        }   

        return centroids;
    }
    

    /**
     * 
    */
    std::vector<MatType> kmeans_single_lloyd(const MatType& X, 
        const VecType& x_squared_norms) const {

        bool converged = false;
        std::size_t num_samples = X.rows(), num_features = X.cols(); 

        MatType centroid;
        centroid = init_centroid(X, x_squared_norms);

        bool converged = false;

        // std::vector<MatType> clusters;

        for (std::size_t iter = 0; i < max_iter_; ++iter) {

            MatType cluster;
            for (std::size_t i = 0; i < num_samples; ++i) {
                double min_dist = ConstType<double>::max();
                std::size_t index = 0;
                // compute the min distance between sample and centroid

                for (std::size_t j = 0; j < num_clusters_; ++j) {
                    double dist = math::norm2<MatType>(X.row(i) - centroid.row(j));

                    if (dist < min_dist) {
                        index = i;
                        min_dist = dist;
                    }

                }

            }

        }

    }


public:
    KMeans(): init_("random"), 
        num_init_(10),
        num_clusters_(8), 
        max_iter_(300) {};

    KMeans(std::string init,
        std::size_t num_init,
        std::size_t num_clusters,
        std::size_t max_iter): init_(init), 
        num_init_(num_init),
        num_clusters_(num_clusters), 
        max_iter_(max_iter) {};


    void test_func(const MatType& X) {
        MatType centroids;

        centroids = init_centroids(X);

        std::cout << centroids << std::endl;
    }





};

} // cluster_model
} // openml

#endif /*METHODS_CLUSTER_KMEANS_HPP*/
