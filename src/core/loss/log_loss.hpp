#ifndef CORE_LOSS_LOG_LOSS_HPP
#define CORE_LOSS_LOG_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class LogLoss {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;
    double reg_coef;
    std::size_t n_features;
    std::string penalty;

public:
    LogLoss(const MatType& X_, 
        const VecType& y_, 
        const double reg_coef_ = 1.0, 
        const std::string penalty_ = "l2"): X(X_), 
            y(y_), reg_coef(reg_coef_) {
                std::size_t n_features = X.cols();
            };

    ~LogLoss() {};


    void shuffle() {
        math::shuffle_data(X, y, X, y);
    };

    double evaluate(const VecType& weight, 
        const std::size_t begin,
        const std::size_t batch_size) const {
        
        MatType X_batch(batch_size, n_features);
        VecType y_batch(batch_size);

        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);

        double retval = 0.0;
        for (std::size_t i = 0; i < batch_size; i++) {
            double val = X_batch.row(i) * weight.transpose();
            retval += (std::log(1 + std::exp(val)) - y_batch(i, 0) * val);
        }

        return retval;
    }

    void gradient(const VecType& weight, 
        const std::size_t begin,
        const std::size_t batch_size, VecType& grad) {
        
        MatType X_batch(batch_size, n_features);
        VecType y_batch(batch_size);

        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);





    }



};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/
