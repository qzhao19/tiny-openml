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
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t p_;
    std::size_t num_neighbors_;
    std::string solver_;
    std::string metric_;

protected:
    void brute_force(const MatType& X, 
        const VecType& y) {
        


        
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