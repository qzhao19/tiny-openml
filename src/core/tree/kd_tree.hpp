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
    using IdxMatType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;
    using NNType = std::pair<DataType, std::size_t>;

    struct neighbor_heap_cmp {
        bool operator()(const NNType &x, const NNType &y) {
            return x.first < y.first;
        }
    };
    using NNHeapType = std::priority_queue<NNType, std::vector<NNType>, neighbor_heap_cmp>;

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
        std::size_t index;
        IdxVecType indices;
        MatType data;
    };

    int ord_;
    MatType data_;
    std::string metric_;
    std::size_t leaf_size_;
    std::shared_ptr<std::vector<KDTreeNode>> tree_;
    
protected:
    const std::size_t find_partition_axis(const MatType& data) const {
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
        tree_ = std::make_shared<std::vector<KDTreeNode>>();
        MatType data = data_;
        std::size_t num_samples = data.rows(), num_features = data.cols();

        // find bounding hyper-rectangle
        MatType hyper_rect(2, num_features);
        hyper_rect.row(0) = data.colwise().minCoeff();
        hyper_rect.row(1) = data.colwise().maxCoeff();

        // create root of kd-tree
        // find the partition axis via function find_partition_axis
        // the sort data along the partition axis 
        std::size_t partition_axis = find_partition_axis(data);
        IdxVecType indices = common::argsort<MatType, IdxMatType, IdxVecType>(
            data.col(partition_axis), 0
        );
        data = data(indices, Eigen::all).eval();
        std::size_t mid_idx = num_samples / 2;
        std::size_t left_num_samples = mid_idx;
        std::size_t right_num_samples = num_samples - mid_idx;
        DataType partition_val = data(mid_idx, partition_axis);

        // define left_hyper_rect and right_hyper_rect
        MatType left_hyper_rect = hyper_rect;
        MatType right_hyper_rect = hyper_rect;
        left_hyper_rect(1, 0) = partition_val;
        right_hyper_rect(0, 0) = partition_val;

        KDTreeNode node; //std::make_shared<>();
        node.left_hyper_rect = std::make_shared<MatType>(left_hyper_rect);
        node.right_hyper_rect = std::make_shared<MatType>(right_hyper_rect);
        tree_->emplace_back(node);

        // create a stack to restore data, indice, parent indice, depth and is_left
        StackData init_stk_data1, init_stk_data2;
        init_stk_data1.is_left = true;
        init_stk_data1.depth = 1;
        init_stk_data1.index = 0;
        init_stk_data1.data = data.topRows(left_num_samples);
        init_stk_data1.indices = indices.head(left_num_samples);

        init_stk_data2.is_left = false;
        init_stk_data2.depth = 1;
        init_stk_data2.index = 0;
        init_stk_data2.data = data.bottomRows(right_num_samples);
        init_stk_data2.indices = indices.tail(right_num_samples);

        std::stack<StackData> node_stk;
        node_stk.push(init_stk_data1);
        node_stk.push(init_stk_data2);

        // recursively split data in halves using hyper-rectangles:
        while (!node_stk.empty()) {
            // pop data off stack
            StackData curr_stk_data = node_stk.top();
            node_stk.pop();

            std::size_t curr_num_samples = curr_stk_data.data.rows();
            std::size_t node_ptr = tree_->size();

            // update parent node index
            KDTreeNode curr_node = tree_->at(curr_stk_data.index);
            if (curr_stk_data.is_left) {
                tree_->at(curr_stk_data.index).left = node_ptr;
            }
            else {
                tree_->at(curr_stk_data.index).right = node_ptr;
            }

            // insert node in kd-tree
            // leaf node ? create leaf node
            if (curr_num_samples <= leaf_size_) {
                KDTreeNode leaf_node;
                leaf_node.data = std::make_shared<MatType>(curr_stk_data.data);
                leaf_node.indices = std::make_shared<IdxVecType>(curr_stk_data.indices);
                leaf_node.left_hyper_rect = nullptr;
                leaf_node.right_hyper_rect = nullptr;
                leaf_node.left = 0;
                leaf_node.right = 0;
                tree_->emplace_back(leaf_node);
            }
            // not a leaf, split the data in two
            else {
                IdxVecType data_indices;
                // split the data in two left_samples and right_samples
                std::size_t curr_mid_idx = curr_num_samples / 2;
                std::size_t curr_left_num_samples = curr_mid_idx;
                std::size_t curr_right_num_samples = curr_num_samples - curr_mid_idx;

                // find the partition axis, sort data along the partition axis
                partition_axis = find_partition_axis(curr_stk_data.data);
                indices = common::argsort<MatType, IdxMatType, IdxVecType>(
                    curr_stk_data.data.col(partition_axis), 0
                );
                data = curr_stk_data.data(indices, Eigen::all).eval();
                data_indices = curr_stk_data.indices(indices);
                partition_val = data(curr_mid_idx, partition_axis);
                node_ptr = tree_->size();

                // push updated data, indices, index and current depth into stack
                StackData update_stk_data1, update_stk_data2;
                update_stk_data1.is_left = true;
                update_stk_data1.depth++;
                update_stk_data1.index = node_ptr;
                update_stk_data1.data = data.topRows(curr_left_num_samples);
                update_stk_data1.indices = data_indices.head(curr_left_num_samples);

                update_stk_data2.is_left = false;
                update_stk_data2.depth++;
                update_stk_data2.index = node_ptr;
                update_stk_data2.data = data.bottomRows(curr_right_num_samples);
                update_stk_data2.indices = data_indices.tail(curr_right_num_samples);

                node_stk.push(update_stk_data1);
                node_stk.push(update_stk_data2);

                if (curr_stk_data.is_left) {
                    left_hyper_rect = *curr_node.left_hyper_rect;
                    right_hyper_rect = *curr_node.left_hyper_rect;
                }
                else {
                    left_hyper_rect = *curr_node.right_hyper_rect;
                    right_hyper_rect = *curr_node.right_hyper_rect;
                }

                left_hyper_rect(1, partition_axis) = partition_val;
                right_hyper_rect(0, partition_axis) = partition_val;

                // create non_leaf node
                KDTreeNode branch_node;
                branch_node.left_hyper_rect = std::make_shared<MatType>(left_hyper_rect);
                branch_node.right_hyper_rect = std::make_shared<MatType>(right_hyper_rect);
                tree_->emplace_back(branch_node);
            }   
        }
    }

    const bool check_intersection(const MatType& hyper_rect, 
        const RowVecType& centroid, 
        double radius) const {

        RowVecType lower_bounds = hyper_rect.row(0);
        RowVecType upper_bounds = hyper_rect.row(1);
        RowVecType c = centroid;

        for (int i = 0; i < c.cols(); ++i) {
            if (c(0, i) > upper_bounds(0, i)) {
                c(0, i) = upper_bounds(0, i);
            }
            if (c(0, i) < lower_bounds(0, i)) {
                c(0, i) = lower_bounds(0, i);
            } 
        }

        double dist = metric::minkowski_distance<RowVecType>(c, centroid, ord_);
        return (dist < radius) ? true : false;
    }

    const std::vector<NNType> compute_distance(const RowVecType& data, 
        const MatType& leaf_data,
        const IdxVecType& leaf_indices, 
        std::size_t k) const {

        std::size_t num_samples = leaf_data.rows();
        if (k >= num_samples) {
            k = num_samples;
        }

        // compute difference between leaf data and single data points
        MatType tmp = leaf_data.rowwise() - data;
        ColVecType dist = common::norm<MatType>(tmp, 1, ord_);
        IdxVecType indices = common::argsort<MatType, IdxMatType, IdxVecType>(dist, 0);
        indices = indices.head(k).eval();

        std::vector<NNType> knn;
        for (int i = 0; i < k; ++i) {
            knn.emplace_back(
                std::make_pair(dist(indices(i), 0), leaf_indices(indices(i)))
            );
        }
        return knn;
    }

    const std::vector<NNType> query_single_data(
        const RowVecType& data, 
        std::size_t k) const {

        std::stack<KDTreeNode> node_stk;
        node_stk.push(tree_->at(0));

        std::vector<NNType> knn(k);
        NNHeapType nn_heap;
        for (std::size_t i = 0; i < k; ++i) {
            nn_heap.push(std::make_pair(ConstType<DataType>::infinity(), 0));
        }

        // recursively iterate over the nodes
        while (!node_stk.empty()) {
            KDTreeNode node = node_stk.top();
            node_stk.pop();
            if (node.indices != nullptr) {
                std::vector<NNType> single_knn = compute_distance(
                    data, *node.data, *node.indices, k
                );
                // push nn into min priority queue 
                for (auto iter = single_knn.begin(); iter != single_knn.end(); iter++) {
                    nn_heap.push(*iter);
                    // if heap size > k, pop top element, keep the size = k
                    if (nn_heap.size() > k) { 
                        nn_heap.pop();
                    }
                }
            }
            else {
                DataType radius = nn_heap.top().first;
                // check left branch
                if (check_intersection(*node.left_hyper_rect, data, radius)) {
                    node_stk.push(tree_->at(node.left));
                }
                // chech right branch
                if (check_intersection(*node.right_hyper_rect, data, radius)) {
                    node_stk.push(tree_->at(node.right));
                }
            }
        }
        // Find the first K smaller elements
        for (int i = k - 1; i >= 0; i--) {
            knn[i] = std::make_pair(nn_heap.top().first, nn_heap.top().second);
            nn_heap.pop();
        }
        return knn;
    }

    const std::vector<NNType> query_radius_single_data(const RowVecType& data,
        double radius) const {
        
        std::stack<KDTreeNode> node_stk;
        node_stk.push(tree_->at(0));
        std::vector<NNType> knn;
        // recursively iterate over the nodes
        while (!node_stk.empty()) {
            KDTreeNode node = node_stk.top();
            node_stk.pop();

            if (node.indices != nullptr) {
                MatType tmp = (*node.data).rowwise() - data;
                ColVecType dist = common::norm<MatType>(tmp, 1, ord_);

                std::vector<Eigen::Index> selected_indices;
                auto selected_dist = (dist.array() <= static_cast<DataType>(radius));
                for (std::size_t i = 0; i < selected_dist.size(); ++i) {
                    if (selected_dist(i, 0)) {
                        selected_indices.push_back(i);
                    }
                }
                if (!selected_indices.empty()) {
                    for (auto idx : selected_indices) {
                        auto index = (*node.indices)(idx);
                        auto distance = dist(idx, 0);
                        knn.emplace_back(std::make_pair(distance, index));
                    }
                }
            }
            else {
                // check left branch
                if (check_intersection(*node.left_hyper_rect, data, radius)) {
                    node_stk.push(tree_->at(node.left));
                }
                // chech right branch
                if (check_intersection(*node.right_hyper_rect, data, radius)) {
                    node_stk.push(tree_->at(node.right));
                }
            }
        }
        return knn;
    }

public:
    KDTree(const MatType& data, 
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

    KDTree(const MatType& data): 
        data_(data), leaf_size_(10), metric_("euclidean") {
            ord_ = 2;
            build_tree();
        };

    ~KDTree() {
        if (tree_ != nullptr) {
            tree_->clear();
        }
    };

    const std::pair<MatType, IdxMatType> query(
        const MatType& data, 
        std::size_t k) const {
        
        std::size_t num_samples = data.rows();
        MatType distances(num_samples, k);
        IdxMatType indices(num_samples, k);

        for(std::size_t i = 0; i < num_samples; ++i) {
            std::vector<NNType> knn;
            knn = query_single_data(data.row(i), k);
            RowVecType dist(k);
            IdxMatType index(1, k);
            for (std::size_t j = 0; j < knn.size(); ++j) {
                dist(0, j) = knn[j].first;
                index(0, j) = knn[j].second;
            }
            distances.row(i) = dist;
            indices.row(i) = index;
        }
        return std::make_pair(distances, indices);
    }

    const std::vector<std::vector<NNType>> query_radius(
        const MatType& data, 
        double radius) const {
        
        std::size_t num_samples = data.rows();
        std::vector<std::vector<NNType>> knn;
        for(std::size_t i = 0; i < num_samples; ++i) {
            std::vector<NNType> nn;
            nn = query_radius_single_data(data.row(i), radius);
            if (!nn.empty()) {
                knn.emplace_back(nn);
            }
        }
        return knn;
    }


};

}
}
#endif /*CORE_TREE_KD_TREE_HPP*/