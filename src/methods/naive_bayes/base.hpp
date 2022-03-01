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
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
protected:
    VecType classes;
    VecType prior_prob;
    std::size_t num_classes;
    std::map<DataType, std::size_t> label_map;

    /**
     * compute log posterior prbabilty,
    */
    virtual const MatType joint_log_likelihood(const MatType& X) const = 0;

    /**
     * Compute the unnormalized prior log probability of y
     * I.e. ``log P(c)`` as an array-like of shape 
     * (n_classes,).
    */
    virtual void get_prior_prob(const VecType& y) {

        std::size_t num_samples = y.rows();
        for (std::size_t i = 0; i < num_samples; i++) {
            label_map[y[i]]++;
        }
        std::size_t i = 0;
        std::vector<DataType> classes_;
        std::vector<DataType> prior_prob_;
        num_classes = 0;
        for (auto &label : label_map) {
            num_classes++;
            classes_.push_back(label.first);
            prior_prob_.push_back(static_cast<DataType>(label.second) / static_cast<DataType>(num_samples));
            i++;
        }
        classes = utils::vec2mat<VecType>(classes_);
        prior_prob = utils::vec2mat<VecType>(prior_prob_);
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
        
        MatType repeated_log_prob_x;
        repeated_log_prob_x = utils::repeat<MatType>(log_prob_x, num_classes, 1);

        return jll.array() - repeated_log_prob_x.array();
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

        IdxType jll_max_idx = utils::argmax<MatType, VecType, IdxType>(jll, 1);
        VecType pred_y(num_samples);

        std::vector<DataType> pred_y_;
        for (std::size_t i = 0; i < num_samples; i++) {
            std::size_t idx = jll_max_idx(i);
            pred_y_.push_back(classes(idx, 0));
        }

        pred_y = utils::vec2mat<VecType>(pred_y_);
        return pred_y;
    }

    /**
     * get the prior prbabilties attributes
    */
    const VecType get_prior_prob() const {
        return prior_prob;
    }

    /**
     * get the classes attributes
    */
    const VecType get_classes() const {
        return classes;
    }

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_BASE_HPP*/
