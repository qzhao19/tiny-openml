#include "../src/core/tree/kd_tree.hpp"   
using namespace openml;


template<typename DataType>
struct KDTreeNode {
    // bool is_leaf;

    DataType value;
    std::size_t axis;
    std::vector<size_t> indices;
    std::shared_ptr<KDTreeNode> left_child;
    std::shared_ptr<KDTreeNode> right_child;
    
    KDTreeNode(DataType value,
        std::size_t axis,
        std::vector<size_t> indices): 
            value(value),
            axis(axis),
            indices(indices),
            left_child(std::shared_ptr<KDTreeNode>()), 
            right_child(std::shared_ptr<KDTreeNode>()) {};

    ~KDTreeNode() {};

    bool is_leaf() {
        
    }


};






