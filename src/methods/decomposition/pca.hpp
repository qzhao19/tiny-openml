#ifndef METHODS_DECOMPOSITION_PCA_HPP
#define METHODS_DECOMPOSITION_PCA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace decomposition {

template<typename DataType>
class PCA {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IndexType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    MatType components;
    VecType explained_var;
    VecType explained_var_ratio;

    std::string solver;
    std::size_t num_components;
    double noise_var;

protected:

    void fit_data(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        if (solver == "svd") {

            MatType U;
            VecType s; 
            MatType V;
            std::tie(U, s, V) = exact_svd<MatType, VecType>(X, false);
            
            // flip eigenvectors' sign to enforce deterministic output
            MatType Vt = V.transpose();
            std::tie(U, Vt) = math::svd_flip(U, Vt);

            explained_var = math::power(s, 2.0) / (static_cast<DataType>(num_samples) - 1.);

            if (num_components > Vt.cols()) {
                throw std::invalid_argument(
                    "Got a invalid parameter 'num_components', "
                    "it must be less than number of features."
                );
            }
            else {
                components = Vt.leftCols(num_components);
            }
        };

        VecType total_var = math::sum<MatType, VecType>(explained_var);
        explained_var_ratio = explained_var / total_var(0, 0);

        if (num_components < std::min(num_samples, num_features)) {
            VecType noise_var_ = math::mean(explained_var.leftCols(num_components));
            noise_var = noise_var_(0, 0);
        }
        else {
            noise_var = 0.0;
        }

    }


public:
    /**
     * Default constructor to create PCA object, linear dimesionality reduction using SVD 
     * of the data to project it to a lower dimesional space, input data shoule be centered 
     * 
     * @param solver the matrix decomnposition policies, 
     *      if svd, will run full svd via arma::svd
     * 
     * @param n_components Number of components to keep
     * @param scale Whether or not to scale the data.
    */
    PCA(): solver("svd"), 
        n_components(2) {};


    PCA(std::string solver_, 
        std::size_t n_components_): 
            solver(solver_), 
            n_components(n_components_) {};

    /**deconstructor*/
    ~PCA() {};







};

} // namespace openml
} // namespace regression

#endif /*METHODS_DECOMPOSITION_PCA_HPP*/
