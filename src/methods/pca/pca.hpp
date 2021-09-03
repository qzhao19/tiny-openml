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
        n_components(2) {};


    PCA(std::string solver_, 
        std::size_t n_components_): 
            solver(solver_), 
            n_components(n_components_) {};
    


protected:
    



private:

    arma::mat components;

    std::string solver;

    std::size_t n_components;
    
};

};
#endif
