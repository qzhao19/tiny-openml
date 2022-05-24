#ifndef METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP
#define METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace mixture_model {

template<typename DataType>
class GaussianMixture {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t max_iter_;
    std::size_t num_components_;
    std::string init_params_;
    std::string covariance_type_;
    double tol_;
    double reg_covar_;


    
    void initialize(const MatType& X) {
        std::size_t num_samples = X.rows();
        MatType resp(num_samples, num_components_);
        if (init_params_ == "random") {
            resp = math::rand<MatType>(num_samples, num_components_);

            VecType sum_resp = math::sum<MatType, VecType>(resp, 1);
            MatType repeated_sum_resp = utils::repeat<MatType>(sum_resp, num_components_, 1);
            resp = resp.array() / repeated_sum_resp.array();
        }

        std::cout << resp << std::endl;

    }



    const std::vector<MatType> estimate_cov_full(const MatType& X, 
        const MatType& resp, 
        const MatType& mean, 
        const VecType& nk, 
        double reg_covar) {
        
        std::size_t num_components = mean.rows(), num_features = mean.cols();

        std::vector<MatType> cov;

        for (std::size_t k = 0; k < num_components; k++) {
            
        }

    }


public:
    GaussianMixture(): max_iter_(100), 
        num_components_(2), 
        init_params_("random"), 
        covariance_type_("full"), 
        tol_(1e-3), 
        reg_covar_(1e-6) {};

    GaussianMixture(std::size_t max_iter,
        std::size_t num_components,
        std::string init_params,
        std::string covariance_type,
        double tol,
        double reg_covar): max_iter_(max_iter), 
            num_components_(num_components), 
            init_params_(init_params), 
            covariance_type_(covariance_type), 
            tol_(tol), 
            reg_covar_(reg_covar) {};



    void test_func(const MatType& X) {

        initialize(X);

    }












};

} // mixture_model
} // openml

#endif /*METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP*/
