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

    MatType components;
    VecType explained_var;
    VecType explained_var_ratio;

    std::string solver;
    std::size_t n_components;
    double noise_var;

protected:
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
