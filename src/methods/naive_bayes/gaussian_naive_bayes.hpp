#ifndef METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP
#define METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace naive_bayes {

template<typename DataType>
class NaiveBayes {

private:
    /**
     * 
    */
    double var_smoothing;
    double alpha;
    std::size_t num_classes;

    



public:
     NaiveBayes(): var_smoothing(1e-9), 
        alpha(0.0) {};


    NaiveBayes(const double var_smoothing_, 
        const double alpha_) :
            var_smoothing(var_smoothing_), 
            alpha(alpha_) {}
    
    
    ~NaiveBayes() {};



};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP*/

