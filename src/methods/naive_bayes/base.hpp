#ifndef METHODS_NAIVE_BAYES_BASE_HPP
#define METHODS_NAIVE_BAYES_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace naive_bayes {

template<typename DataType>
class BaseNB {

private:

    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
protected:
    VecType classes_;
    VecType prior_prob_;
    std::size_t num_classes_;
    std::map<DataType, std::size_t> label_map_;

    /**
     * compute log posterior prbabilty,
    */
    virtual const MatType joint_log_likelihood(const MatType& X) const = 0;

    /**
     * Compute the unnormalized prior log probability of y
     * I.e. ``log P(c)`` as an array-like of shape 
     * (n_classes,).
    */
    virtual void compute_prior_prob(const VecType& y) {
        std::size_t num_samples = y.rows();
        for (std::size_t i = 0; i < num_samples; i++) {
            label_map_[y[i]]++;
        }
        std::size_t i = 0;
        num_classes_ = label_map_.size();
        
        prior_prob_.resize(num_classes_);
        classes_.resize(num_classes_);
        for (auto &label : label_map_) {
            classes_(i, 0) = label.first;
            prior_prob_(i, 0) = static_cast<DataType>(label.second) / static_cast<DataType>(num_samples);
            ++i;
        }
    }

public:
    BaseNB() {};
    ~BaseNB() {};

    /**
     * Return log-probability estimates for the test data X.
     * @param X ndarray of shape (num_samples, num_features)
     *      input dataset
     * @return array-like of shape (num_samples, num_classes)
     *      the log-probability of the samples for each class in
            the model.
    */
    const MatType predict_log_prob(const MatType& X) const{
        std::size_t num_samples = X.rows(), num_features = X.cols();
        MatType jll;
        jll = this->joint_log_likelihood(X);
        
        // normalize by P(x) = P(f_1, ..., f_n)
        VecType log_prob_x;
        log_prob_x = math::logsumexp<MatType, VecType>(jll, 1);
        
        MatType tmp;
        tmp = common::repeat<MatType>(log_prob_x, num_classes_, 1);

        return jll.array() - tmp.array();
    }

    /**
     * Return probability for the test data X.
     * @param X ndarray of shape (num_samples, num_features)
     *      input dataset
     * @return array-like of shape (num_samples, num_classes)
     *      Returns the probability of the samples for each class
     *      The columns correspond to the classes in sorted order
    */
    const MatType predict_prob(const MatType& X) const{
        MatType log_prob = predict_log_prob(X);
        return log_prob.array().exp();
    }

    /**
     * Perform classification on an array of test data X.
     * @param X ndarray of shape (num_samples, num_features)
     *      input dataset
     * @return array-like of shape (num_samples)
     *      Predicted target values for X.
    */
    const VecType predict(const MatType& X) const {
        std::size_t num_samples = X.rows();
        MatType jll;
        jll = this->joint_log_likelihood(X);

        IdxVecType argmax_jll = common::argmax<MatType, VecType, IdxVecType>(jll, 1);
        VecType y_pred = argmax_jll.template cast<DataType>();
        return y_pred;
    }

    /**
     * get the prior prbabilties attributes
    */
    const VecType get_prior_prob() const {
        return prior_prob_;
    }

    /**
     * get the classes attributes
    */
    const VecType get_classes() const {
        return classes_;
    }

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_BASE_HPP*/
