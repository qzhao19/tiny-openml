#ifndef METHODS_CLUSTER_KMEANS_HPP
#define METHODS_CLUSTER_KMEANS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace cluster {

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


};

} // cluster_model
} // openml

#endif /*METHODS_CLUSTER_KMEANS_HPP*/
