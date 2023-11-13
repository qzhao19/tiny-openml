#ifndef CORE_TREE_BALL_TREE_HPP
#define CORE_TREE_BALL_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

template<typename DataType>
class BallTree {   
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVecType = Eigen::Matrix<DataType, 1, Eigen::Dynamic>;
    using ColVecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    using IdxMatType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;
    using NNType = std::pair<DataType, std::size_t>;

    int ord_;
    MatType data_;
    std::string metric_;
    std::size_t leaf_size_;
    std::shared_ptr<std::vector<KDTreeNode>> tree_;
    

public:
    BallTree(const MatType& data, 
        std::size_t leaf_size, std::string metric): data_(data), 
            leaf_size_(leaf_size), metric_(metric) {
        if (metric == "manhattan") {
            ord_ = 1;
        }
        else if (metric == "euclidean") {
            ord_ = 2;
        }
        else if (metric == "chebyshev") {
            ord_ = Eigen::Infinity;
        }

        build_tree();
    };

    BallTree(const MatType& data): 
        data_(data), leaf_size_(10), metric_("euclidean") {
            ord_ = 2;
            build_tree();
        };

    ~BallTree() {
        if (tree_ != nullptr) {
            tree_->clear();
        }
    };

};

}
}
#endif /*CORE_TREE_KD_TREE_HPP*/