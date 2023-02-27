#ifndef METHOD_NEIGHBORS_KDTREE_HPP
#define METHOD_NEIGHBORS_KDTREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "./base.hpp"
using namespace openml;

namespace openml {
namespace neighbors {

template<typename DataType>
class KDTree {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t p_;
    std::size_t num_neighbors_;
    std::string metric_;

    



public:
    KDTree(): num_neighbors_(15), 
        p_(1), 
        metric_("minkowski") {};
    

    KDTree(std::size_t num_neighbors, 
        std::size_t p, 
        std::string metric): num_neighbors_(num_neighbors), 
            p_(p), 
            metric_(metric) {};

};

} // neighbors
} // openml

#endif /*METHOD_NEIGHBORS_KDTREE_HPP*/