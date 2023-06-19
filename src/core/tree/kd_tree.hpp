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
void free_tree_memory(tree_node *root) {
    std::stack<tree_node *> node_stack;
    tree_node *p;
    node_stack.push(root);
    while (!node_stack.empty()) {
        p = node_stack.top();
        node_stack.pop();
        if (p->left)
            node_stack.push(p->left);
        if (p->right)
            node_stack.push(p->right);
        free(p);
    }
}


inline KDTree::KDTree(tree_node *root, const float *datas, size_t rows, size_t cols, float p) :
        root(root), datas(datas), n_samples(rows),
        n_features(cols), p(p), free_tree_(false) {
    InitBuffer();
    labels = nullptr;
}

inline KDTree::KDTree(const float *datas, const float *labels, size_t rows, size_t cols, float p, bool free_tree) :
        datas(datas), labels(labels), n_samples(rows), n_features(cols), p(p), free_tree_(free_tree) {
    std::vector<size_t> points;
    for (size_t i = 0; i < n_samples; ++i)
        points.emplace_back(i);
    InitBuffer();
    root = BuildTree(points);
}

inline KDTree::~KDTree() {
    delete[]get_mid_buf_;
    delete[]visited_buf_;
#ifdef USE_INTEL_MKL
    free(mkl_buf_);
#endif
    if (free_tree_)
        free_tree_memory(root);
}


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

std::tuple<size_t, float> KDTree::MidElement(const std::vector<size_t> &points, size_t dim) {
    size_t len = points.size();
    for (size_t i = 0; i < points.size(); ++i)
        get_mid_buf_[i] = std::make_tuple(points[i], GetDimVal(points[i], dim));
    std::nth_element(get_mid_buf_, 
                     get_mid_buf_ + len / 2,
                     get_mid_buf_ + len,
                     [](const std::tuple<size_t, float> &i, const std::tuple<size_t, float> &j) {
                        return std::get<1>(i) < std::get<1>(j);
                     });
    return get_mid_buf_[len / 2];
}

inline float KDTree::GetDist(size_t i, const float *coor) {
    float dist = 0.0;
    size_t idx = i * n_features;
    for (int t = 0; t < n_features; ++t)
        dist += pow(datas[idx + t] - coor[t], p);
    return static_cast<float>(pow(dist, 1.0 / p));
}


inline void KDTree::InitBuffer() {
    get_mid_buf_ = new std::tuple<size_t, float>[n_samples];
    visited_buf_ = new bool[n_samples];

#ifdef USE_INTEL_MKL
    // 要与 C 代码交互，所以用 C 的方式申请内存
    mkl_buf_ = Malloc(float, n_features);
#endif
}

inline void KDTree::HeapStackPush(std::stack<tree_node *> &paths, tree_node *node, const float *coor, size_t k) {
    paths.emplace(node);
    size_t id = node->id;
    if (visited_buf_[id])
        return;
    visited_buf_[id] = true;
    float dist = GetDist(id, coor);
    std::tuple<size_t, float> t(id, dist);
    if (k_neighbor_heap_.size() < k)
        k_neighbor_heap_.push(t);

    else if (std::get<1>(t) < std::get<1>(k_neighbor_heap_.top())) {
        k_neighbor_heap_.pop();
        k_neighbor_heap_.push(t);
    }
}

tree_model *build_kdtree(const float *datas, const float *labels,
                         size_t rows, size_t cols, float p) {
    KDTree tree(datas, labels, rows, cols, p, false);
    tree_model *model = Malloc(tree_model, 1);
    model->datas = datas;
    model->labels = labels;
    model->n_features = cols;
    model->n_samples = rows;
    model->root = tree.GetRoot();
    model->p = p;
    return model;
}

// 求平均值，用于回归问题
float mean(const float *arr, size_t len) {
    float ans = 0.0;
    for (size_t i = 0; i < len; ++i)
        ans += arr[i];
    return ans / len;
}


}
}
#endif /*CORE_TREE_KD_TREE_HPP*/