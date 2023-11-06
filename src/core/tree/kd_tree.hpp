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
    // using NNType = std::pair<ColVecType, IdxVecType>;
    // using NNType = std::vector<std::pair<DataType, std::size_t>>;
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
    std::string metric_;
    std::size_t leaf_size_;
    std::shared_ptr<std::vector<KDTreeNode>> tree_;
    MatType data_;

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
                partition_val = data(curr_mid_idx, partition_axis);
                node_ptr = tree_->size();

                // push updated data, indices, index and current depth into stack
                StackData update_stk_data1, update_stk_data2;
                update_stk_data1.is_left = true;
                update_stk_data1.depth++;
                update_stk_data1.index = node_ptr;
                update_stk_data1.data = data.topRows(curr_left_num_samples);
                update_stk_data1.indices = indices.head(curr_left_num_samples);

                update_stk_data2.is_left = false;
                update_stk_data2.depth++;
                update_stk_data2.index = node_ptr;
                update_stk_data2.data = data.bottomRows(curr_right_num_samples);
                update_stk_data2.indices = indices.tail(curr_right_num_samples);

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

        std::vector<NNType> knn(k, std::make_pair(ConstType<DataType>::infinity(), 0));
        NNHeapType nn_heap;
        // std::vector<NNType> knn;
        // knn.emplace_back(std::make_pair(ConstType<DataType>::infinity(), 0));
        // std::vector<NNType> nns(
        //     k, std::make_pair(ConstType<DataType>::infinity(), 0)
        // );
        // for (std::size_t i = 0; i < k; ++i) {
        //     nn_heap.push(std::make_pair(ConstType<DataType>::infinity(), 0));
        // }
        // while(!nn_heap.empty()){
        //     std::cout << nn_heap.top().first << " " << nn_heap.top().second << std::endl;
        //     nn_heap.pop();
        // }

        while (!node_stk.empty()) {
            KDTreeNode node = node_stk.top();
            node_stk.pop();

            for (const auto &nn : knn) {
                std::cout << nn.first << " " << nn.second << " ";
            }
            std::cout << std::endl;

            // bool found = node.indices != nullptr ? true : false;
            // std::cout << "left_hyper_rect:" << found <<std::endl;
            // if (node.left_hyper_rect) {
            //     std::cout << "left_hyper_rect:" << std::endl;
            //     std::cout << *node.left_hyper_rect << std::endl;
            // }

            if (node.indices != nullptr) {
                // std::cout << "left_hyper_rect:" << std::endl;
                std::vector<NNType> single_knn = compute_distance(
                    data, *node.data, *node.indices, k
                );

                // for (size_t i = 0; i < single_knn.size(); i++) {
                //     nn_heap.push(single_knn[i]);
                // }
                // while (k > 0) {
                //     knn.emplace_back(nn_heap.top());
                //     k--;
                // }
                for (auto iter = single_knn.begin(); iter != single_knn.end(); iter++) {
                    nn_heap.push(*iter);
                    // if heap size > k, pop top element, keep the size = k
                    if (nn_heap.size() > k) { 
                        nn_heap.pop();
                    }
                }
                // Find the first K smaller elements
                for (int i = k - 1; i >= 0; i--) {
                    knn[i] = std::make_pair(nn_heap.top().first, nn_heap.top().second);
                    nn_heap.pop();
                }
            }
            else {
                DataType radius = knn.back().first;
                if (check_intersection(*node.left_hyper_rect, data, radius)) {
                    node_stk.push(tree_->at(node.left));
                }

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
    };

    KDTree(const MatType& data): 
        data_(data), leaf_size_(10), metric_("euclidean") {
            ord_ = 2;
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

        // std::cout << "num_samples = " << num_samples << std::endl;

        for(std::size_t i = 0; i < num_samples; ++i) {
            std::vector<NNType> knn;
            knn = query_single_data(data.row(i), k);

            // for (const auto &nn : knn) {
            //     std::cout << nn.first << " " << nn.second << std::endl;
            // }

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


    void test(const MatType& data) {

        // std::size_t axis = find_partition_axis(data_);
        // std::cout << data << std::endl;

        build_tree();
        std::cout << "build_tree" << std::endl;

        MatType distances;
        IdxMatType indices;

        std::tie(distances, indices) = query(data, 4);

        std::cout << "distances" << std::endl;
        std::cout << distances << std::endl;

        std::cout << "indices" << std::endl;
        std::cout << indices << std::endl;

        // for (auto node : *tree_) {
        //     if (node.left_hyper_rect) {
        //         std::cout << "left_hyper_rect" << std::endl;
        //         std::cout << *node.left_hyper_rect << std::endl;
        //     }

        //     if (node.right_hyper_rect) {
        //         std::cout << "right_hyper_rect" << std::endl;
        //         std::cout << *node.right_hyper_rect << std::endl;
        //     }

        //     if (node.data) {
        //         std::cout << "data" << std::endl;
        //         std::cout << *node.data << std::endl;
        //     }

        //     if (node.indices) {
        //         std::cout << "indices" << std::endl;
        //         std::cout << *node.indices << std::endl;
        //     }

        //     std::cout << "left = " << node.left << ", right = " << node.right << std::endl;
        // }

        // MatType hyper_rect{{4.3, 2.0 , 1.0 , 0.1},{4.4, 4.4, 6.9, 2.5}};
        // RowVecType centroid{{4.4, 3.5, 1.4, 0.2, 0.9}};
        // RowVecType c{{5.1, 3.58, 1.7, 0.24, 1.8}};

        // std::cout << metric::minkowski_distance<RowVecType>(c, centroid, 2) << std::endl;
        // std::cout << metric::minkowski_distance<RowVecType>(c, centroid, 1) << std::endl;


        // RowVecType data{{7.2, 3.2, 6.0, 1.8}};
        // MatType leaf_data{{7.1, 3.0, 5.9, 2.1},
        //                   {6.3, 3.3, 6.0, 2.5},
        //                   {7.2, 3.6, 6.1, 2.5},
        //                   {7.3, 2.9, 6.3, 1.8},
        //                   {7.6, 3.0, 6.6, 2.1},
        //                   {7.7, 3.8, 6.7, 2.2},
        //                   {7.7, 2.8, 6.7, 2.0},
        //                   {7.7, 2.6, 6.9, 2.3}};

        // IdxVecType leaf_indices(8); leaf_indices<<102, 100, 109, 107, 105, 117, 122, 118;

        // NNType nn = compute_distance(data, leaf_data, leaf_indices, 5); 

        // for (auto n : nn) {
        //     std::cout << n.first << " " << n.second << std::endl;
        // }

        // std::cout << res.first << std::endl;
        // std::cout << res.second << std::endl;

        // std::cout << leaf_indices;


    };



};


}
}
#endif /*CORE_TREE_KD_TREE_HPP*/