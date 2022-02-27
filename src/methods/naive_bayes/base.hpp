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
    VecType prior_prob;
    std::size_t num_classes;
    std::map<DataType, std::size_t> label_map;

    /**
     * 
    */
    virtual const MatType joint_log_likelihood(const MatType& X) const = 0;

    /**
     * compute the class prior probability P(y)
    */
    virtual void get_prior_prob(const VecType& y) {

        std::size_t num_samples = y.rows();
        for (std::size_t i = 0; i < num_samples; i++) {
            label_map[y[i]]++;
        }
        std::size_t i = 0;
        std::vector<double> prior_prob_;
        num_classes = 0;
        for (auto &label : label_map) {
            num_classes++;
            prior_prob_.push_back(static_cast<DataType>(label.second) / static_cast<DataType>(num_samples));
            i++;
        }
        prior_prob = utils::vec2mat<VecType>(prior_prob_);
    }

    


public:
    BaseNB() {};
    ~BaseNB() {};

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_BASE_HPP*/
