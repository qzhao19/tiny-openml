#ifndef CORE_TREE_KD_TREE_HPP
#define CORE_TREE_KD_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

template<typename DataType>
class KDTree {
private:
    struct KDTreeNode {
        int left;
        int right;
        std::shared_ptr<std::vector<std::size_t>> indices;
        std::shared_ptr<std::vector<std::vector<DataType>>> data;
        std::shared_ptr<std::vector<std::vector<DataType>>> left_hyper_rect;
        std::shared_ptr<std::vector<std::vector<DataType>>> right_hyper_rect;

        KDTreeNode(): left(-1), 
            right(-1), 
            indices(nullptr), 
            data(nullptr), 
            left_hyper_rect(nullptr), 
            right_hyper_rect(nullptr) {};

        KDTreeNode(int left_, 
            int right_, 
            std::shared_ptr<std::vector<std::size_t>> indices_, 
            std::shared_ptr<std::vector<std::vector<DataType>>> data_, 
            std::shared_ptr<std::vector<std::vector<DataType>>> left_hyper_rect_, 
            std::shared_ptr<std::vector<std::vector<DataType>>> right_hyper_rect_): 
                left(left_), 
                right(right_), 
                data(data_), 
                left_hyper_rect(left_hyper_rect_), 
                right_hyper_rect(right_hyper_rect_) {};
        
        ~KDTreeNode() {};
    };

    struct StackData {
        bool is_left;
        std::size_t depth;
        std::size_t parent;
        std::vector<std::size_t> indices;
        std::vector<std::vector<DataType>> data;
    };

    std::size_t leaf_size_;
    std::vector<KDTreeNode> tree_;
    const std::vector<std::vector<DataType>> data_;
    
protected:
    std::size_t find_partition_axis(const std::vector<std::vector<DataType>>& data) {
        std::size_t num_samples = data.size();
        std::size_t num_features = data[0].size();
        std::vector<DataType> range_bounds(num_features);
        std::vector<DataType> lower_bounds(num_features, ConstType<DataType>::infinity());
        std::vector<DataType> upper_bounds(num_features, -ConstType<DataType>::infinity());

        for (std::size_t i = 0; i < num_samples; ++i) {
            for (std::size_t j = 0; j < num_features; ++j) {
                lower_bounds[j] = std::min(lower_bounds[j], data[i][j]);
                upper_bounds[j] = std::max(upper_bounds[j], data[i][j]);
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

        std::vector<std::vector<DataType>> data = data_;
        std::size_t num_samples = data.size();
        std::size_t num_features = data[0].size();
        
        // find bounding hyper-rectangle
        std::vector<DataType> lower_bounds(num_features, ConstType<DataType>::infinity());
        std::vector<DataType> upper_bounds(num_features, -ConstType<DataType>::infinity());

        for (std::size_t i = 0; i < num_samples; ++i) {
            for (std::size_t j = 0; j < num_features; ++j) {
                lower_bounds[j] = std::min(lower_bounds[j], data[i][j]);
                upper_bounds[j] = std::max(upper_bounds[j], data[i][j]);
            }
        }

        std::vector<std::vector<DataType>> hyper_rect(2, std::vector<DataType>(num_features));
        hyper_rect[0].emplace_back(lower_bounds);
        hyper_rect[1].emplace_back(upper_bounds);

        // create root of kd-tree
        std::size_t partition_axis = find_partition_axis(data);
        std::vector<DataType> partition_data;

        for (const auto& row : data) {
            partition_data.emplace_back(row[partition_axis]);
        }
        std::vector<std::size_t> indices = common::argsort<DataType>(partition_data);
        
        for (std::size_t i = 0; i < indices.size(); ++i) {
            data[i] = data[indices[i]];
        }

        std::size_t mid_idx = num_samples / 2;
        DataType partition_val = data[mid_idx][partition_axis];

        std::vector<std::vector<DataType>> left_hyper_rect, right_hyper_rect;
        left_hyper_rect = hyper_rect;
        right_hyper_rect = hyper_rect;
        left_hyper_rect[1][0] = partition_val;
        right_hyper_rect[0][0] = partition_val;

        KDTreeNode node; //std::make_shared<NodeType>();
        node.left_hyper_rect = left_hyper_rect;
        node.right_hyper_rect = right_hyper_rect;

        tree_.emplace_back(node);

        // create a stack to restore data, indice, parent indice, depth and is_left
        StackData s1, s2;
        s1.is_left = true;
        s1.depth = 1;
        s1.parent = 0;
        s1.data = common::slice<DataType>(data, 0, mid_idx, 0, num_features - 1);
        s1.indices = common::slice<DataType>(indices, 0, mid_idx);

        s2.is_left = false;
        s2.depth = 1;
        s2.parent = 0;
        s2.data = common::slice<DataType>(data, mid_idx, num_samples - 1, 0, num_features - 1);
        s2.indices = common::slice<DataType>(indices, mid_idx, num_samples - 1);

        std::stack<StackData> stack;
        stack.push(s1);
        stack.push(s2);

        // recursively split data in halves using hyper-rectangles:
        while (!stack.empty()) {
            // pop data off stack
            auto tmp_stack = stack.top();
            stack.pop();

            num_samples = tmp_stack.data.size();
            std::size_t node_ptr = tree_.size();

            // update parent node
            KDTreeNode tmp_node;
            tmp_node = tree_[tmp_stack.parent];

            if (s.is_left) {
                KDTreeNode left_node;
                left_node.left = node_ptr;
                left_node.right = tmp_node.right;
                left_node.indices = tmp_node.indices;
                left_node.data = tmp_node.data;
                left_node.left_hyper_rect = tmp_node.left_hyper_rect;
                left_node.right_hyper_rect = tmp_node.right_hyper_rect;
                tree_[tmp_stack.parent] = left_node;
            }
            else {
                KDTreeNode right_node;
                right_node.left = tmp_node.left;
                right_node.right = node_ptr;
                right_node.indices = tmp_node.indices;
                right_node.data = tmp_node.data;
                right_node.left_hyper_rect = tmp_node.left_hyper_rect;
                right_node.right_hyper_rect = tmp_node.right_hyper_rect;
                tree_[tmp_stack.parent] = right_node;
            }

            // check leaf node, if yes, insert node in kd-tree
            if (num_samples < leaf_size_) {
                KDTreeNode leaf;
                leaf.data = tmp_stack.data;


            }




        }

    }
    


public:
    KDTree(const std::vector<std::vector<DataType>>& data, 
        std::size_t leaf_size): data_(data), leaf_size_(leaf_size) {};

    KDTree(const std::vector<std::vector<DataType>>& data): 
        data_(data), leaf_size_(10) {};

    ~KDTree() {
        data_.clear();
    };


    void test() {

        std::size_t axis = find_partition_axis(data_);

        std::cout << axis << std::endl;

        build_tree();

    };



};


}
}
#endif /*CORE_TREE_KD_TREE_HPP*/