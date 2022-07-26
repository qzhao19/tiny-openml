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
        MatType centroid(num_clusters_, num_features);

        if (init_ == "random") {
            IdxType index = math::permutation<IdxType>(num_samples);
            IdxType selected_index = index.topRows(num_clusters_);
            centroid = X(selected_index, Eigen::all);
        }   

        return centroid;
    }
    

    /**
     * k-means lloyd
    */
    MatType kmeans_lloyd(const MatType& X, 
        const MatType& centroid) const {

        bool converged = false;
        std::size_t num_samples = X.rows(), num_features = X.cols(); 
        MatType new_centroid(num_clusters_, num_features);

        for (std::size_t iter = 0; iter < max_iter_; ++iter) {
            // define the cluster
            std::map<std::size_t, std::vector<std::size_t>> cluster;
            
            for (std::size_t i = 0; i < num_samples; ++i) {
                std::size_t min_dist_index = 0;
                double min_dist = ConstType<double>::max();
                // compute the min distance between sample and centroid
                
                for (std::size_t j = 0; j < num_clusters_; ++j) {
                    MatType tmp;
                    tmp = X.row(i) - centroid.row(j);
                    double dist = static_cast<double>(tmp.norm());
                    if (dist < min_dist) {
                        min_dist_index = j;
                        min_dist = dist;
                    }
                    
                }
                cluster[min_dist_index].push_back(i);
            }

            for (auto& c : cluster) {
                std::size_t num = c.second.size();
                IdxType v(num);
                for (std::size_t i = 0; i < num; ++i) {
                    v(i) = c.second[i];
                }
                std::cout << "min_dist_index = " << c.first << " vect = " << v.transpose() << std::endl;
            }
            



            MatType old_centroid = centroid;
            for (auto& c : cluster) {
                std::size_t num_samples_cluster = c.second.size();
                MatType sample(1, num_features);
                sample.setZero();
                for (std::size_t i = 0; i < num_samples_cluster; ++i) {
                    sample = sample.array() + X(c.second[i], Eigen::all).array();
                }
                MatType mean_sample;
                mean_sample = sample.array() / static_cast<DataType>(num_samples_cluster);
                new_centroid.row(c.first) = mean_sample;
            }

            // std::cout << "old_centroid = " << old_centroid << std::endl;
            // std::cout << "new_centroid = " << new_centroid << std::endl;
            // auto diff = (old_centroid - new_centroid).norm();
            // std::cout << "iter = " << iter << ", matrix norm = " << diff << std::endl;
            // MatType diff;
            // if (!old_centroid.isApprox(new_centroid)) {

                // std::cout << "iter = " << iter << std::endl;
                // break; 
            // }
        }
        return new_centroid;
    }


public:
    KMeans(): init_("random"), 
        num_init_(10),
        num_clusters_(3), 
        max_iter_(5) {};

    KMeans(std::string init,
        std::size_t num_init,
        std::size_t num_clusters,
        std::size_t max_iter): init_(init), 
        num_init_(num_init),
        num_clusters_(num_clusters), 
        max_iter_(max_iter) {};


    void test_func(const MatType& X) {
        MatType centroids;

        centroids = init_centroid(X, VecType());

        std::cout << "centroids" << std::endl;
        std::cout << centroids << std::endl;


        MatType new_centroids;
        new_centroids = kmeans_lloyd(X, centroids);

        std::cout << "new_centroids" << std::endl;
        std::cout << new_centroids << std::endl;
    }





};

} // cluster_model
} // openml

#endif /*METHODS_CLUSTER_KMEANS_HPP*/
