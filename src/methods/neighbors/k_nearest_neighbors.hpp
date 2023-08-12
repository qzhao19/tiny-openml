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
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t p_;
    std::size_t num_neighbors_;
    std::string solver_;
    std::string metric_;

    std::vector<std::vector<DataType>> X_{};  
    std::vector<DataType> y_{};                  

protected:
    int brute_force(std::vector<DataType>& sample, int k) {
        
        std::vector<int> neighbors;
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < this->X_.size(); ++i) {
            auto current = this->X_.at(i);
            auto label = this->Y_.at(i);
            auto distance = euclidean_distance(current, sample);
            distances.emplace_back(distance, label);
        }
        std::sort(distances.begin(), distances.end());
        for (int i = 0; i < k; i++) {
            auto label = distances.at(i).second;
            neighbors.push_back(label);
        }
        std::unordered_map<int, int> frequency;
        for (auto neighbor : neighbors) {
            ++frequency[neighbor];
        }
        std::pair<int, int> predicted;
        predicted.first = -1;
        predicted.second = -1;
        for (auto& kv : frequency) {
            if (kv.second > predicted.second) {
                predicted.second = kv.second;
                predicted.first = kv.first;
            }
        }
        return predicted.first;
    }



public:
    KNearestNeighbors(): num_neighbors_(15), 
        p_(1), 
        solver_("brute"),
        metric_("minkowski") {};
    

    KNearestNeighbors(std::size_t num_neighbors, 
        std::size_t p, 
        std::string solver, 
        std::string metric): num_neighbors_(num_neighbors), 
            p_(p), 
            solver_(solver),
            metric_(metric) {};

};

} // neighbors
} // openml

#endif /*METHOD_NEIGHBORS_K_NEAREST_NEIGHBORS_HPP*/