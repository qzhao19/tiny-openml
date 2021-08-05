#ifndef METHOD_PERCEPTRON_PERCEPTRON_HPP
#define METHOD_PERCEPTRON_PERCEPTRON_HPP
#include "../../core.hpp"
#include "../../prereqs.hpp"
#include "weight_initializer/zero_initializer.hpp"


namespace perceptron {

template<typename WeightInitializer>
class Perceptron {

public:
    /**
     * Default constructor, create the perceptron with given parameters having 
     * default values.
     * 
     * @param initializer Weights initializer to initialize w, string with default value 
     *                    zerosInitializer
     * @param shuffle Whether or not the training data should be shuffled after each epoch
     * @param alpha learning rate when update parameters, default = 0.001
     * @param tol The stopping criterion, default = 1e-3 
     * @param max_iter The maximum number of passes over the training data, default=1000
    */
    Perceptron(): initializer("zeros"), 
        shuffle(true),
        alpha(0.01), 
        tol(1e-5), 
        max_iter(10000) {};
    

    Perceptron(const std::string initializer_,
        const bool shuffle_,
        const double alpha_, 
        const double tol_, 
        const std::size_t max_iter_): initializer(initializer_), 
            shuffle(shuffle_),
            alpha(alpha_), 
            tol(tol_), 
            max_iter(max_iter_) {};


    
    
    
protected:
    void fit(const arma::mat &X, 
             const arma::vec &y) const;

    double sign(const arma::vec& x, 
                const arma::vec& w, 
                const double b) const;

private:
    arma::vec weights;
    double bias;

    std::string initializer;
    bool shuffle;
    double alpha;
    double tol;
    std::size_t max_iter;




};

};
#endif
