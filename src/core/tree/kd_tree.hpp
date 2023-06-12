#ifndef CORE_TREE_KD_TREE_HPP
#define CORE_TREE_KD_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

struct tree_node
{
    std::size_t id;
    std::size_t split;
    tree_node *left, *right;
};


struct tree_model
{
    tree_node *root;
    const float *datas;
    const float *labels;
    size_t n_samples;
    size_t n_features;
    float p;
};


class KDTree {
public:
	KDTree(){}

    KDTree(tree_node *root, const float *datas, size_t rows, size_t cols, float p);

    KDTree(const float *datas, const float *labels,
           size_t rows, size_t cols, float p, bool free_tree = true);

    ~KDTree();

    tree_node *GetRoot() { return root; }

    std::vector<std::tuple<size_t, float>> FindKNearests(const float *coor, size_t k);

    std::tuple<size_t, float> FindNearest(const float *coor, size_t k) { return FindKNearests(coor, k)[0]; }

    void CFindKNearests(const float *coor, size_t k, size_t *args, float *dists);


private:
    // The sample with the largest distance from point `coor`
    // is always at the top of the heap.
    struct neighbor_heap_cmp {
        bool operator()(const std::tuple<size_t, float> &i,
                        const std::tuple<size_t, float> &j) {
            return std::get<1>(i) < std::get<1>(j);
        }
    };

    typedef std::tuple<size_t, float> neighbor;
    typedef std::priority_queue<neighbor,
            std::vector<neighbor>, neighbor_heap_cmp> neighbor_heap;

    
    neighbor_heap k_neighbor_heap_;
    float p;
    bool free_tree_;
    tree_node *root;
    const float *datas;
    size_t n_samples;
    size_t n_features;
    const float *labels;
    std::tuple<size_t, float> *get_mid_buf_;
    bool *visited_buf_;

    void InitBuffer();
    tree_node *BuildTree(const std::vector<size_t> &points);
    std::tuple<size_t, float> MidElement(const std::vector<size_t> &points, size_t dim);
    void HeapStackPush(std::stack<tree_node *> &paths, tree_node *node, const float *coor, size_t k);
    float GetDimVal(size_t sample, size_t dim) {
        return datas[sample * n_features + dim];
    }
    float GetDist(size_t i, const float *coor);

    size_t FindSplitDim(const std::vector<size_t> &points);

};


std::size_t KDTree::FindSplitDim(const std::vector<size_t> &points) {
    if (points.size() == 1)
        return 0;
    size_t cur_best_dim = 0;
    float cur_largest_spread = -1;
    float cur_min_val;
    float cur_max_val;
    for (size_t dim = 0; dim < n_features; ++dim) {
        cur_min_val = GetDimVal(points[0], dim);
        cur_max_val = GetDimVal(points[0], dim);
        for (const auto &id : points) {
            if (GetDimVal(id, dim) > cur_max_val)
                cur_max_val = GetDimVal(id, dim);
            else if (GetDimVal(id, dim) < cur_min_val)
                cur_min_val = GetDimVal(id, dim);
        }

        if (cur_max_val - cur_min_val > cur_largest_spread) {
            cur_largest_spread = cur_max_val - cur_min_val;
            cur_best_dim = dim;
        }
    }
    return cur_best_dim;
}

}
}
#endif /*CORE_TREE_KD_TREE_HPP*/