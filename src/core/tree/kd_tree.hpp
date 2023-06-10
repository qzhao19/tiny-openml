#ifndef CORE_TREE_KD_TREE_HPP
#define CORE_TREE_KD_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

template<typename NodeType, typename DataType>
class KDTree {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t leaf_size_;
    std::string metric_;
    std::shared_ptr<NodeType> root_;
    
    

public:
    KDTree(): leaf_size_(40), 
        metric_("minkowski") {};
    

    KDTree(std::size_t leaf_size, 
        std::string metric): leaf_size_(leaf_size), 
            metric_(metric) {};
    

};

}
}
#endif /*CORE_TREE_KD_TREE_HPP*/