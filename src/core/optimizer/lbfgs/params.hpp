#ifndef CORE_OPTIMIZER_LBFGS_PARAMS_HPP
#define CORE_OPTIMIZER_LBFGS_PARAMS_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

template<typename DataType = double>
class LinearSearchParams {
public:
    DataType dec_factor_;
    DataType inc_factor_;
    DataType ftol_;
    DataType wolfe_;
    DataType max_step_;
    DataType min_step_;
    std::size_t max_linesearch_;
    std::string condition_;

public:
    LinearSearchParams(): dec_factor_(0.5), 
        inc_factor_(2.1), 
        ftol_(1e-4), 
        wolfe_(0.9), 
        max_step_(1e+20), 
        min_step_(1e-20), 
        max_linesearch_(40), 
        condition_("WOLFE") {};

    LinearSearchParams(DataType dec_factor,
        DataType inc_factor,
        DataType ftol,
        DataType wolfe,
        DataType max_step,
        DataType min_step,
        std::size_t max_linesearch,
        std::string condition): dec_factor_(dec_factor), 
            inc_factor_(inc_factor), 
            ftol_(ftol), 
            wolfe_(wolfe), 
            max_step_(max_step), 
            min_step_(min_step), 
            max_linesearch_(max_linesearch), 
            condition_(condition) {};

};

}
}
#endif /*CORE_OPTIMIZER_LBFGS_PARAMS_HPP*/
