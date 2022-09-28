#ifndef METHODS_LINEAR_MODEL_PERCEPTRON_HPP
#define METHODS_LINEAR_MODEL_PERCEPTRON_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class Perceptron: public BaseLinearModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    
protected:

    bool shuffle_;
    bool verbose_;
    double alpha_;
    double tol_;
    double lambda_;
    std::size_t batch_size_;
    std::size_t max_iter_;
    std::string penalty_;

    /**fit data implementation*/
    void fit_data(const MatType& X, 
        const VecType& y) {
    };

public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    Perceptron(): BaseLinearModel<DataType>(true), 
        shuffle_(true), 
        verbose_(false), 
        alpha_(0.001), 
        lambda_(0.5), 
        tol_(0.0001), 
        batch_size_(32), 
        max_iter_(1000), 
        penalty_("l2") {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param lambda: The penalty (aka regularization term) to be used. 
     *      Constant that multiplies the regularization term
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
    */
    Perceptron(bool intercept,
        bool shuffle, 
        bool verbose, 
        const double alpha, 
        const double lambda,
        const double tol, 
        const std::size_t batch_size, 
        const std::size_t max_iter, 
        const std::string penalty): 
            BaseLinearModel<DataType>(intercept), 
            shuffle_(shuffle), 
            verbose_(verbose), 
            alpha_(alpha), 
            lambda_(lambda), 
            tol_(tol), 
            batch_size_(batch_size), 
            max_iter_(max_iter), 
            penalty_(penalty_) {};

    /**deconstructor*/
    ~Perceptron() {};

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_PERCEPTRON_HPP*/
