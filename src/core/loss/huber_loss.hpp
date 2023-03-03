#ifndef CORE_LOSS_HUBER_LOSS_HPP
#define CORE_LOSS_HUBER_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class HuberLoss {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double delta_;
    // double threshold_;

public:
    HuberLoss(): delta_(1.0) {};
    HuberLoss(double delta): delta_(delta){};
    ~HuberLoss() {};

    

};

}
}
#endif /*CORE_LOSS_HUBER_LOSS_HPP*/
