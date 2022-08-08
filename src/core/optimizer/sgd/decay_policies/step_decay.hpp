#ifndef CORE_OPTIMIZER_SGD_DECAY_POLICIES_STEP_DECAY_HPP
#define CORE_OPTIMIZER_SGD_DECAY_POLICIES_STEP_DECAY_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType>
class StepDecay {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double base_lr_;
    double gamma_;
    std::size_t step_size_;

public:
    StepDecay(): base_lr_(0.1), 
            gamma_(0.5), 
            step_size_(10){};

    explicit StepDecay(double base_lr, 
        double gamma = 0.5, 
        std::size_t step_size = 10): base_lr_(base_lr), 
            gamma_(gamma), 
            step_size_(step_size){};
    
    ~StepDecay() {};

    /**
     * lr = base_lr * gamma ^ floor(epoch / step_size) 
    */
    double compute(std::size_t epoch) {
        std::size_t warmup_iters = (1 + epoch) / step_size_;
        double lr = base_lr_ * std::pow(gamma_, warmup_iters);
        return lr;
    }
};

}
}
#endif /*CORE_OPTIMIZER_SGD_DECAY_POLICIES_STEP_DECAY_HPP*/
