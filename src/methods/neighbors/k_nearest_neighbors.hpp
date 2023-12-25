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

    MatType X_;
    ColVecType y_;
    std::unique_ptr<tree::KDTree<DataType>> tree_;

protected:
    DataType brute_force(const MatType& X, const ColVecType& y, const RowVecType& samples) {
        
        std::vector<DataType> neighbors;
        std::vector<std::pair<double, DataType>> lookup_dist;

        for (std::size_t i = 0; i < X.row(); ++i) {
            RowVecType current = X.rows(i);
            DataType label = y(i, 0);
            double distance;
            
            if (metric_ == "euclidean") {
                distance = (current - sample).template lpNorm<2>();
            }
            else if (metric_ == "manhattan") {
                distance = (current - sample).template lpNorm<1>();
            }
            else {
                std::ostringstream err_msg;
                err_msg << "Only support euclidean and manhattan distance." << std::endl;
                throw std::invalid_argument(err_msg.str());

            }
            lookup_dist.emplace_back(distance, label);
        }
        std::sort(lookup_dist.begin(), lookup_dist.end());
        for (std::size_t i = 0; i < num_neighbors_; i++) {
            auto label = lookup_dist.at(i).second;
            neighbors.push_back(label);
        }
        std::unordered_map<DataType, std::size_t> frequency;
        for (auto neighbor : neighbors) {
            ++frequency[neighbor];
        }
        std::pair<DataType, std::size_t> pred_label;
        pred_label.first = -1;
        pred_label.second = -1;
        for (auto& kv : frequency) {
            if (kv.second > pred_label.second) {
                pred_label.second = kv.second;
                pred_label.first = kv.first;
            }
        }
        return pred_label.first;

    }
      

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
        
        if (solver_ == "kdtree") {
            // call kdTree via a function pionter
            tree_ = std::make_unique<tree::KDTree<DataType>>(
                X, leaf_size_, metric_
            );
            
        }
        else if (solver_ == "brute") {
            X_ = X;
        }
        y_ = y;
        
    }

    const ColVecType predict(const MatType& X) {
        ColVecType pred_label;
        if (solver_ == "kdtree") {
            MatType neighbor_dist;
            IdxMatType neighbor_indice;
            std::tie(neighbor_dist, neighbor_indice) = tree_->query(X, num_neighbors_);

            std::size_t num_samples = X.rows();
            MatType pred_indice(num_samples, num_neighbors_);
            for (std::size_t i = 0; i < num_samples; ++i) {
                pred_indice.row(i) = y_(neighbor_indice.row(i), Eigen::all).transpose();
            }

            MatType mode, count;
            std::tie(mode, count) = math::mode<MatType>(pred_indice, 0);

            pred_label = mode;
        }
        else if (solver_ == "brute") {
            std::size_t num_samples = X.rows();
            pred_label.resize(num_samples);
            for (std::size_t i = 0; i < num_samples; ++i) {
                DataType lable = brute_force(X_, y_, X.row(i));
                pred_label(i, 0) = lable;
            }
        }
        
        return pred_label;
    }

};

} // neighbors
} // openml

#endif /*METHOD_NEIGHBORS_K_NEAREST_NEIGHBORS_HPP*/