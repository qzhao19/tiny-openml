#ifndef METHOD_PCA_PCA_HPP
#define METHOD_PCA_PCA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

#include "decomposition_policies/eig_method.hpp"
#include "decomposition_policies/exact_svd_method.hpp"

namespace pca{

class PCA {
public:

    /**
     * Default constructor to create PCA object, linear dimesionality reduction using SVD 
     * of the data to project it to a lower dimesional space, input data shoule be centered 
     * 
     * @param solver the matrix decomnposition policies, if eig, will run eigen vector
     * decomposition, if svd, will run full svd via arma::svd
     * 
     * @param n_components Number of components to keep
     * @param scale Whether or not to scale the data.
    */
    PCA(): solver("full_svd"), 
        n_components(2), 
        scale(false) {};


    PCA(std::string solver_, 
        std::size_t n_components_, 
        bool scale_): 
            solver(solver_), 
            n_components(n_components_), 
            scale(scale_) {};
    
    /**deconstructor*/
    ~PCA() {};

    /**
     * Apply Principal Component Analysis to the provided data set.
     * @param X Dataset on which training should be performed
    */
    void fit(const arma::mat& X);

    /**
     * Apply dimensionality reduction to X.
    */
    const arma::mat transform(const arma::mat& X); 

    /**
     * Return the average log-likelihood of all samples.
    */
    const double score(const arma::mat& X);

protected:
    /**
     * Using SVD method 
    */
    template<typename DecompositionPolicy>
    const arma::mat svd_train(const arma::mat& X, 
        DecompositionPolicy& decomposition_policy) const;

    const arma::mat eig_train(const arma::mat& X) const;

    /**
     * Scaling the data is when we reduce the variance of each dimension to 1.
    */
    void scale_data(arma::mat& X) const;

    /**
     * Return the log-likelihood of each sample.
    */
    const arma::vec score_samples(const arma::mat& X) const;


private:

    arma::mat components;

    arma::rowvec explained_var;

    arma::rowvec explained_var_ratio;

    std::string solver;

    std::size_t n_components;

    bool scale;
    
};

};
#endif
