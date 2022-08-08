#ifndef CORE_OPTIMIZER_SGD_DECAY_POLICIES_EXPONENTIAL_DECAY_HPP
#define CORE_OPTIMIZER_SGD_DECAY_POLICIES_EXPONENTIAL_DECAY_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType>
class ExponentialDecay {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double base_lr_;
    double gamma_;

public:
    ExponentialDecay(): base_lr_(0.1), 
            gamma_(0.5) {};

    explicit ExponentialDecay(double base_lr, 
        double gamma = 0.5): base_lr_(base_lr), 
            gamma_(gamma) {};
    
    ~ExponentialDecay() {};

    /**
     * lr = base_lr * exp(-gamma * epoch)
    */
    double compute(std::size_t epoch) {
        double lr = base_lr_ * std::exp(gamma_ * static_cast<double>(epoch));
        return lr;
    }

};

}
}
#endif /*CORE_OPTIMIZER_SGD_DECAY_POLICIES_EXPONENTIAL_DECAY_HPP*/
