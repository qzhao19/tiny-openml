#ifndef METHOD_NEIGHBORS_KD_TREE_HPP
#define METHOD_NEIGHBORS_KD_TREE_HPP
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
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t p_;
    std::size_t num_neighbors_;
    std::string metric_;

    struct Node {
        bool is_leaf;
        IdxVecType data_indices_per_node;
        std::weak_ptr<Node> parent;
        std::shared_ptr<Node> left_child;
        std::shared_ptr<Node> right_child;

        Node():is_leaf(true),
            data_indices_per_node(IdxVecType()),
            parent(std::weak_ptr<Node>()),
            left_child(std::shared_ptr<Node>()), 
            right_child(std::shared_ptr<Node>()) {};

        Node(bool is_leaf_,
            IdxVecType data_indices_per_node_): 
            is_leaf(is_leaf_),
            data_indices_per_node(data_indices_per_node_),
                parent(std::weak_ptr<Node>()),
                left_child(std::shared_ptr<Node>()), 
                right_child(std::shared_ptr<Node>()) {};

        ~Node(){};
    };



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

#endif /*METHOD_NEIGHBORS_KD_TREE_HPP*/
