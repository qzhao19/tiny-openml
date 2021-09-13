#include "pca.hpp"

using namespace math;
using namespace pca;

void PCA::scale_data(arma::mat& X) {
    arma::vec scaled_vec = arma::stddev(X, 0, 1);
    // if there are any zero, make them small

    for (std::size_t i = 0; i < scaled_vec.n_elem; i++) {
        if (scaled_vec[i] == 0) {
            scaled_vec[i] = 1e-30;
        }
    }
    X = arma::repmat(scaled_vec, 1, X.n_cols);
    
}

template<typename DecompositionPolicy>
const arma::mat PCA::train(const arma::mat& X, 
        DecompositionPolicy& decomposition_policy) const {
    
    


}



