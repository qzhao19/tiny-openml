#ifndef CORE_TREE_KD_TREE_HPP
#define CORE_TREE_KD_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

template<typename DataType>
class KDTree {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVecType = Eigen::Matrix<DataType, 1, Eigen::Dynamic>;
    using ColVecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;


    struct KDTreeNode {
        int left;
        int right;
        std::shared_ptr<MatType> data;
        std::shared_ptr<IdxVecType> indices;
        std::shared_ptr<MatType> left_hyper_rect;
        std::shared_ptr<MatType> right_hyper_rect;

        KDTreeNode(): left(-1), 
            right(-1), 
            data(nullptr),
            indices(nullptr), 
            left_hyper_rect(nullptr), 
            right_hyper_rect(nullptr) {};

        KDTreeNode(int left_, 
            int right_, 
            std::shared_ptr<MatType> data_, 
            std::shared_ptr<IdxVecType> indices_,
            std::shared_ptr<MatType> left_hyper_rect_, 
            std::shared_ptr<MatType> right_hyper_rect_): 
                left(left_), 
                right(right_), 
                data(data_), 
                indices(indices_), 
                left_hyper_rect(left_hyper_rect_), 
                right_hyper_rect(right_hyper_rect_) {};
        
        ~KDTreeNode() {};
    };

    struct StackData {
        bool is_left;
        std::size_t depth;
        std::size_t parent;
        IdxVecType indices;
        MatType data;
    };

    std::size_t leaf_size_;
    std::vector<KDTreeNode> tree_;
    MatType data_;
    
protected:
    std::size_t find_partition_axis(const MatType& data) {
        std::size_t num_samples = data.rows(), num_features = data.cols();
        std::vector<DataType> range_bounds(num_features);
        std::vector<DataType> lower_bounds(num_features, ConstType<DataType>::infinity());
        std::vector<DataType> upper_bounds(num_features, -ConstType<DataType>::infinity());

        for (std::size_t i = 0; i < num_samples; ++i) {
            for (std::size_t j = 0; j < num_features; ++j) {
                lower_bounds[j] = std::min(lower_bounds[j], data(i, j));
                upper_bounds[j] = std::max(upper_bounds[j], data(i, j));
            }
        }

        std::size_t partition_axis = 0;
        for (std::size_t i = 0; i < num_features; ++i) {
            range_bounds[i] = std::abs(upper_bounds[i] - lower_bounds[i]);
            if (range_bounds[i] > range_bounds[partition_axis]) {
                partition_axis = i;
            }
        }

        return partition_axis;
    };


    void build_tree() {
        MatType data = data_;
        std::size_t num_samples = data.rows(), num_features = data.cols();

        // find bounding hyper-rectangle
        MatType hyper_rect(2, num_features);
        hyper_rect.row(0) = data.colwise().min();
        hyper_rect.row(1) = data.colwise().max();

        // create root of kd-tree
        // find the partition axis via function find_partition_axis
        // the sort data along the partition axis 
        std::size_t partition_axis = find_partition_axis(data);
        IdxVecType indices = common::argsort(data.col(partition_axis), 1);
        data = data(indices, Eigen::all);

        std::size_t mid_idx = num_samples / 2;
        DataType partition_val = data(mid_idx, partition_axis);

        // define left_hyper_rect and right_hyper_rect
        MatType left_hyper_rect = hyper_rect;
        MatType right_hyper_rect = hyper_rect;
        left_hyper_rect(1, 0) = partition_val;
        right_hyper_rect(0, 0) = partition_val;

        KDTreeNode node; //std::make_shared<>();
        node.left_hyper_rect = std::make_shared<MatType>(left_hyper_rect);
        node.right_hyper_rect = std::make_shared<MatType>(right_hyper_rect);
        tree_.emplace_back(node);

        // create a stack to restore data, indice, parent indice, depth and is_left
        StackData s1, s2;
        s1.is_left = true;
        s1.depth = 1;
        s1.parent = 0;
        s1.data = data.topRows(mid_idx);
        s1.indices = indices.head(mid_idx);

        s2.is_left = false;
        s2.depth = 1;
        s2.parent = 0;
        s1.data = data.bottomRows(mid_idx);
        s1.indices = indices.tail(mid_idx);

        std::stack<StackData> stack;
        stack.push(s1);
        stack.push(s2);

        // recursively split data in halves using hyper-rectangles:
        while (!stack.empty()) {
            // pop data off stack
            auto s = stack.top();
            stack.pop();

            std::size_t cur_num_samples = s.data.rows()
            std::size_t node_ptr = tree_.size();

            // update parent node
            KDTreeNode parent = tree_[s.parent];
            if (s.is_left) {
                KDTreeNode left_child;
                left_child.left = node_ptr;
                left_child.right = parent.right;
                left_child.indices = parent.indices;
                left_child.data = parent.data;
                left_child.left_hyper_rect = parent.left_hyper_rect;
                left_child.right_hyper_rect = parent.right_hyper_rect;
                tree_[s.parent] = left_child;
            }
            else {
                KDTreeNode right_child;
                right_child.left = parent.right;
                right_child.right = node_ptr;
                right_child.indices = parent.indices;
                right_child.data = parent.data;
                right_child.left_hyper_rect = parent.left_hyper_rect;
                right_child.right_hyper_rect = parent.right_hyper_rect;
                tree_[s.parent] = right_child;
            }

            // insert node in kd-tree
            // leaf node ? create leaf node
            if (cur_num_samples < leaf_size_) {
                KDTreeNode leaf_node;
                
            }
            
        }
    }


public:
    KDTree(const MatType& data, 
        std::size_t leaf_size): data_(data), leaf_size_(leaf_size) {};

    KDTree(const MatType& data): 
        data_(data), leaf_size_(10) {};

    ~KDTree() {
        tree_.clear();
    };


    void test() {

        std::size_t axis = find_partition_axis(data_);

        std::cout << axis << std::endl;

        // build_tree();

    };



};


}
}
#endif /*CORE_TREE_KD_TREE_HPP*/