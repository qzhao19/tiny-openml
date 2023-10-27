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

    struct NodeStackData {
        bool is_left;
        std::size_t depth;
        std::size_t index;
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
        NodeStackData init_stk_data1, init_stk_data2;
        init_stk_data1.is_left = true;
        init_stk_data1.depth = 1;
        init_stk_data1.index = 0;
        init_stk_data1.data = data.topRows(mid_idx);
        init_stk_data1.indices = indices.head(mid_idx);

        init_stk_data2.is_left = false;
        init_stk_data2.depth = 1;
        init_stk_data2.index = 0;
        init_stk_data2.data = data.bottomRows(mid_idx);
        init_stk_data2.indices = indices.tail(mid_idx);

        std::stack<NodeStackData> node_stk;
        node_stk.push(init_stk_data1);
        node_stk.push(init_stk_data2);

        // recursively split data in halves using hyper-rectangles:
        while (!node_stk.empty()) {
            // pop data off stack
            NodeStackData curr_stk_data = node_stk.top();
            node_stk.pop();

            std::size_t curr_num_samples = curr_stk_data.data.rows();
            std::size_t node_ptr = tree_.size();

            // update parent node index
            if (curr_stk_data.is_left) {
                tree_[curr_stk_data.index].left = node_ptr;
            }
            else {
                tree_[curr_stk_data.index].right = node_ptr;
            }

            // insert node in kd-tree
            // leaf node ? create leaf node
            if (curr_num_samples < leaf_size_) {
                KDTreeNode leaf_node;
                leaf_node.data = std::make_shared<MatType>(curr_stk_data.data);
                leaf_node.indices = std::make_shared<IdxVecType>(curr_stk_data.indices);
                leaf_node.left_hyper_rect = nullptr;
                leaf_node.right_hyper_rect = nullptr;
                leaf_node.left = 0;
                leaf_node.right = 0;
                tree_.emplace_back(leaf_node);
            }
            // not a leaf, split the data in two
            else {
                partition_axis = find_partition_axis(curr_stk_data.data);
                indices = common::argsort(curr_stk_data.data.col(partition_axis), 1);
                data = curr_stk_data.data(indices, Eigen::all);
                node_ptr = tree_.size();

                NodeStackData update_stk_data1, update_stk_data2;
                update_stk_data1.is_left = true;
                update_stk_data1.depth = 1;
                update_stk_data1.index = 0;
                update_stk_data1.data = data.topRows(mid_idx);
                update_stk_data1.indices = indices.head(mid_idx);

                update_stk_data2.is_left = false;
                update_stk_data2.depth = 1;
                update_stk_data2.index = 0;
                update_stk_data2.data = data.bottomRows(mid_idx);
                update_stk_data2.indices = indices.tail(mid_idx);



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