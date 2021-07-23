#ifndef METHOD_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
#define METHOD_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
#include "logistic_regression_function.hpp"
#include "../../prereqs.hpp"


namespace regression {

class LogisticRegression {
public:

    /**
     * Default constructor of logistic regression. it could custom optimizer
     * parameters. 
     * 
     * @param solver Algorithms to use in optimization of objective function,
     *               default value is SGD method
     * @param intercept Specifies if a intercept term should be added to the 
     *                  decision function.
     * @param lambda L2 regularization coefficient, small values specify stronger 
     *               regularization.
     * @param alpha  Learning rate when update weights
     * @param tol    Tolerance for stopping criteria.
     * @param batch_size Number of samples that will be passed through the optimizer
     *                   specifies if we apply sgd method
     * @param max_iter Maximum number of iterations taken for the solvers to converge.
     * @param n_basis  Number of memory points to be stored (default 10).
     * 
    */
    LogisticRegression(): solver("sgd"), 
                          shuffle(true),
                          lambda(0.1), 
                          alpha(0.01), 
                          tol(1e-5),
                          batch_size(32), 
                          max_iter(10000), 
                          n_basis(10) {}; 

    /**
     * Non-empty constructor, create the model with params
    */
    LogisticRegression(const std::string solver_,
                       const bool shuffle_ = true,
                       const double lambda_ = 0.1, 
                       const double alpha_ = 0.01, 
                       const double tol_ = 1e-5, 
                       const std::size_t batch_size_ = 32, 
                       const std::size_t max_iter_ = 10000, 
                       const std::size_t n_basis_ = 10): 
        solver(solver_), 
        shuffle(shuffle_),
        lambda(lambda_), 
        alpha(alpha_), 
        tol(tol_), 
        batch_size(batch_size_), 
        max_iter(max_iter_), 
        n_basis(n_basis_) {};


    ~LogisticRegression() {};

    /**
     * Train the logistic regression model on given dataset. By default, 
     * the sgd optimizer is used. The overload function of fit(...)
     * 
     * @param X Input dataset matrix for training 
     * @param y Vector of label associated the dataset
     * 
    */
    void fit(const arma::mat &X, const arma::vec &y);

    /**
     * Predict class labels for samples in X
     * 
     * @param X Input dataset matrix for training 
    */
    const arma::vec predict(const arma::mat &X) const;

    /**
     * Probability estimates. The returned estimates for all classes are 
     * ordered by the label of classes.
     * 
     * @param X Input dataset matrix for training 
     * 
     * @return y_pred_prob Probability of the sample for each class in the model, 
     *         ndarray of shape (n_samples, n_classes)
     * 
    */
    const arma::mat predict_prob(const arma::mat &X) const;

    /**
     * Accuracy classification score. calculate the mean accuracy 
     * on the given lable of test dataset and predicted label.
     * 
     * @param y_true True label for dataset X, ndarray of shape (n_samples,)
     * @param y_pred Predicted label from predict(), ndarray of shape (n_samples,)
     * 
     * @return acc Mean accuracy of y' and y.
    */
    const double score(const arma::vec &y_true, 
                       const arma::vec &y_pred) const;
    
    /**
     * Return the training params theta, 
    */
    const arma::vec& get_theta() const { return theta; };

protected:
    /**
     * Train the logistic model with the given optimizer, using the overload allow
     * configuring the optimizer before training.
     * 
     * @param X Input training dataset, ndarray of shape [n_samples, n_features]
     * @param y Vector of asscociated label with dataset, ndnarry of shape [m_samples]
     * @param OptimizerType Type of optimizer to use to train the model
     * @param CallbackTypes Types of Callback Functions.
    */
    template<typename OptimizerType, typename... CallbackTypes>
    const arma::vec fit(const arma::mat& X, 
                        const arma::vec& y, 
                        OptimizerType& optimizer,
                        CallbackTypes&&... callbacks) const;

private:
    /**
     * @param theta Vector of logistic regression function parameters
     * @param solver Algorithm to use in the optimization problem, default is 
     *               sgd optimizer. There are two methods to optimize loss funct
     * @param intercept Whether to fit the intercept for training the model. 
     *                  default = True
     * @param lambda L2 regularization parameter, default is 0.0
     * @param alpha  learning rate when update weights
     * @param batch_size Batch size to use for each step, when use SGD optimizer
     * @param max_iter Maximum number of iterations for the optimization, default is 100000
     * @param tol Tolence for stopping criteria, support only SGD optimizer, 
     *            default is 1e-5
     * @param shuffle If true, the function order is shuffled; otherwise, each function 
     *                is visited in linear order. default is True
     * @param n_basis Number of memory points to be stored (default 10).
    */
    arma::vec theta;
    
    std::string solver;

    bool shuffle;

    double lambda;

    double alpha;

    double tol;

    std::size_t batch_size;

    std::size_t max_iter;

    std::size_t n_basis;
};

};

#endif
