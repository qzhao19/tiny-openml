#ifndef METHOD_NEIGHBORS_K_NEAREST_NEIGHBORS_HPP
#define METHOD_NEIGHBORS_K_NEAREST_NEIGHBORS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "./base.hpp"
using namespace openml;

namespace openml {
namespace neighbors {

template<typename DataType>
class KNearestNeighbors {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVecType = Eigen::Matrix<DataType, 1, Eigen::Dynamic>;
    using ColVecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    using IdxMatType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;
    using NNType = std::pair<DataType, std::size_t>;

    std::size_t leaf_size_;
    std::size_t num_neighbors_;
    std::string solver_;
    std::string metric_;

    ColVecType y_;
    std::unique_ptr<tree::KDTree<DataType>> tree_;       

public:
    KNearestNeighbors(): leaf_size_(10),
        num_neighbors_(15),  
        solver_("kdtree"),
        metric_("euclidean") {};
    

    KNearestNeighbors(std::size_t leaf_size, 
        std::size_t num_neighbors,
        std::string solver, 
        std::string metric): leaf_size_(leaf_size), 
            num_neighbors_(num_neighbors),
            solver_(solver),
            metric_(metric) {};

    
    void fit(const MatType& X, const ColVecType& y) {
        // call kdTree via a function pionter
        tree_ = std::make_unique<tree::KDTree<DataType>>(
            X, leaf_size_, metric_
        );
        y_ = y;
    }

    void predict(const MatType& X) {
        MatType neighbor_dist;
        IdxMatType neighbor_ind;
        std::tie(neighbor_dist, neighbor_ind) = tree_->query(X, num_neighbors_);

        std::size_t num_samples = X.rows();
        MatType pred_ind(num_samples, num_neighbors_);
        for (std::size_t i = 0; i < num_samples; ++i) {
            pred_ind.row(i) = y_(neighbor_ind.row(i), Eigen::all).transpose();
        }


    
    }

};

} // neighbors
} // openml

#endif /*METHOD_NEIGHBORS_K_NEAREST_NEIGHBORS_HPP*/