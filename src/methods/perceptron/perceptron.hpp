#ifndef METHOD_PERCEPTRON_PERCEPTRON_HPP
#define METHOD_PERCEPTRON_PERCEPTRON_HPP
#include "../../core.hpp"
#include "../../prereqs.hpp"
#include "weight_initializer/ones_initializer.hpp"


namespace perceptron {

template<typename WeightInitializer = OnesInitializer>
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
    Perceptron(): initializer("ones"), 
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

    

    /**
     * Train the perceptron on the given data for up to the 
     * maximum number of iterations. A single iteration 
     * corresponds to a single pass through the data.
     * 
     * @param X Dataset on which training should be performed
     * @param y Labels of the dataset.
    */
    void fit(const arma::mat &X, 
        const arma::vec &y) const;


    /**
     * Classification function. After training, use the weights 
     * matrix to classify test dataset 
     * 
     * @param X shape of [n_samples, n_features] input testing dataset
     * 
    */
    const arma::mat predict(const arma::mat &X) const;


    /**
     * 
    */
    const double Perceptron::score(const arma::vec &y_true, 
        const arma::vec &y_pred) const;

protected:
    /**
     * 
    */
    double sign(const arma::rowvec& x, 
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
