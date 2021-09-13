#ifndef METHOD_PCA_PCA_HPP
#define METHOD_PCA_PCA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace pca{

class PCA {
public:

    /**
     * 
    */
    PCA(): solver("exact_svd"), 
        n_components(2), 
        scale(false) {};


    PCA(std::string solver_, 
        std::size_t n_components_, 
        bool scale_): 
            solver(solver_), 
            n_components(n_components_), 
            scale(scale_) {};
    
    ~PCA() {};

protected:
    /**
     * 
    */
    template<typename DecompositionPolicy>
    const arma::mat train(const arma::mat& X, 
        DecompositionPolicy& decomposition_policy) const;

    /**
     * Scaling the data is when we reduce the variance of each dimension to 1.
    */
    void scale_data(arma::mat& X);


private:
    arma::mat components;

    std::string solver;

    std::size_t n_components;

    bool scale;
};
};
#endif
