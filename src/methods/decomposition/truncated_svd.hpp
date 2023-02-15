#ifndef METHODS_DECOMPOSITION_TRUNCATED_SVD_HPP
#define METHODS_DECOMPOSITION_TRUNCATED_SVD_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace decomposition {

/**
 * linear dimesionality reduction using SVD of the data to project it 
 * to a lower dimesional space, input data shoule be centered 
 * 
 * @param solver the matrix decomnposition policies, 
 *      if svd, will run full svd via math::exact_svd
 * 
 * @param n_components Number of components to keep
 * @param scale Whether or not to scale the data.
*/
template<typename DataType>
class TruncatedSVD {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    
};

} // namespace openml
} // namespace regression

#endif /*METHODS_DECOMPOSITION_PCA_HPP*/
