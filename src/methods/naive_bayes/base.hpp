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

protected:
    VecType prior_prob;
    std::size_t num_classes;
    std::map<DataType, std::size_t> label_map;

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
        for (auto &label : label_map) {
            num_classes++;
            prior_prob_.push_back(static_cast<DataType>(label.second) / static_cast<DataType>(num_samples));
            i++;
        }
    }


public:
    BaseNB(const VecType& prior_prob_, 
        const std::map<DataType, std::size_t>& label_map_): num_classes(0), 
            prior_prob(prior_prob_), 
            label_map(label_map_) {};

    ~BaseNB() {};

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_BASE_HPP*/
