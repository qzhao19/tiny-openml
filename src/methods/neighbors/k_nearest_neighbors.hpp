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

protected:
    DataType brute_force(const std::vector<std::vector<DataType>>& X, 
        const std::vector<DataType>& y, 
        const std::vector<DataType>& sample) {
        
        std::vector<DataType> neighbors;
        std::vector<std::pair<DataType, DataType>> distances;
        for (std::size_t i = 0; i < X.size(); ++i) {
            auto current = X.at(i);
            auto label = y.at(i);
            auto distance;
            if (metric_ == "minkowski") {
                distance = metric::euclidean_distance<DataType>(current, sample);
            }
            else if (metric_ == "") {
                distance = metric::manhattan_distance<DataType>(current, sample);
            }

            distances.emplace_back(distance, label);
        }
        std::sort(distances.begin(), distances.end());
        for (std::size_t i = 0; i < num_neighbors_; i++) {
            auto label = distances.at(i).second;
            neighbors.push_back(label);
        }
        std::unordered_map<DataType, std::size_t> frequency;
        for (auto neighbor : neighbors) {
            ++frequency[neighbor];
        }
        std::pair<DataType, DataType> predicted;
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