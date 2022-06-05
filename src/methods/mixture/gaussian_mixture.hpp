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

    std::vector<MatType> precisions_cholesky_;


    /**
     * Compute the Cholesky decomposition of the precisions
     * @param covariances : ndarray of shape [n, m, k]
     *      The covariance matrix of the current components.
     * 
     * @return precisions_cholesky : array-like cholesky
     *      The cholesky decomposition of sample precisions 
     *      of the current components.
    */
    const std::vector<MatType> compute_precision_cholesky(
        const std::vector<MatType>& covariances) const {
        
        std::size_t num_features = covariances[0].rows();

        std::vector<MatType> precision_chol;
        for (std::size_t i = 0; i < num_components_; ++i) {
            MatType cov_chol;
            cov_chol = math::cholesky<MatType>(covariances[i], true);

            MatType b(num_features, num_features);
            b.setIdentity();

            MatType precision;
            precision = cov_chol.template triangularView<Eigen::Lower>().transpose().solve(b);

            precision_chol.push_back(precision);
        }

        return precision_chol;
    }
    
    /**
     * Compute the log-det of the cholesky decomposition of matrices
     * Given matrix A, cholesky factorization of A is LL^T, 
     * log_det = 2 * sum(log(L.diagonal)) 
     * 
     * @param covariances : array-like of [n_components, n_features, n_features]
     *      Cholesky decompositions of the matrices.
     * @param num_features : int 
     *      Number of features.
     * @return log_det_precision_chol, The determinant of the precision 
     *      matrix for each component
    */
    const VecType compute_log_det_cholesky(
        const std::vector<MatType>& matrix_chol, 
        std::size_t num_features) const {
        
        VecType log_det_chol(num_components_);
        for (std::size_t i = 0; i < num_components_; ++i) {
            VecType diag = matrix_chol[i].diagonal();
            auto tmp = math::sum<MatType, VecType>(diag.array().log(), 0);
            log_det_chol(i) = tmp.value();
        }

        return log_det_chol;
    }


    /**
     * Estimate the log Gaussian probability.
     * The determinant of the precision matrix from the Cholesky decomposition
     * corresponds to the negative half of the determinant of the full precision
     * matrix.
     * 
    */
    const MatType estimate_log_gaussian_prob(
        const std::vector<MatType>& precision_chol,
        const MatType& X, 
        const MatType& means) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_components = means.rows();

        VecType log_det = compute_log_det_cholesky(precisions_cholesky_, num_features);

        MatType log_prob(num_samples, num_components);

        for (std::size_t i = 0; i < num_components; ++i) {
            MatType prec_chol = precision_chol[i];
            VecType mu = means.row(i).transpose();

            MatType tmp1(num_samples, num_features);
            tmp1 = X * prec_chol;

            MatType tmp2(1, num_features);
            tmp2 = mu * prec_chol;

            


        }



    }






    /**
     * Estimate the full covariance matrices.
     * @param X ndarray of shape [num_samples, num_features]
     *      The input data array.
     * @param resp, ndarray of shape [num_samples, num_components]
     *      The responsibilities for each data sample in X.
     * @param means ndarray of shape [num_components, num_feature]
     *      The centers of the current components.
     * @param nk, ndarray of shape [num_components]
     *      The numbers of data samples in the current components.
     * @param  reg_covar : float
     *      The regularization added to the diagonal of the 
     *      covariance matrices.
     * @return covariances : array-like 
     *      The covariance matrix of the current components.
     * 
    */
    const std::vector<MatType> estimate_cov_full(const MatType& X, 
        const MatType& resp, 
        const MatType& means, 
        const VecType& nk, 
        double reg_covar) const {
        
        std::size_t num_samples = X.rows();
        std::size_t num_components = means.rows(), num_features = means.cols();
        std::vector<MatType> covariances;

        for (std::size_t k = 0; k < num_components; k++) {
            MatType diff(num_samples, num_features);
            for (std::size_t i = 0; i < num_samples; ++i) {
                diff.row(i) = X.row(i).array() - means.row(k).array();
            }

            MatType resp_tmp = utils::repeat<MatType>(resp.col(k), num_features, 1);
            // matrix product dot 
            MatType resp_diff_tmp = resp_tmp.array() * diff.array();
            MatType resp_diff = resp_diff_tmp.transpose() * diff;
            
            MatType nk_tmp(num_features, num_features);
            DataType val = nk(k);
            nk_tmp.fill(val);

            // compute covariance for each component
            MatType cov = resp_diff.array() / nk_tmp.array(); 
            cov = cov.array() + reg_covar;
            covariances.push_back(cov);
        }
        return covariances;
    }

    /**
     * Estimate the Gaussian distribution parameters.
     * @param X ndarray of shape (n_samples, n_feature)
     *      The input data array
     * @param resp ndarray of shape (n_samples, n_components)
     *      the responsibilities for each data sample in X.
     * @return nk ndarray of shape (n_components,) 
     *      The numbers of data samples in the current components.
     * @return means : array-like of shape (n_components, n_features)
    */
    std::tuple<std::vector<MatType>, MatType, VecType> 
    estimate_gaussian_parameters(const MatType& X, 
        const MatType& resp, 
        double reg_covar, 
        std::string cov_type = "full") {
        
        std::size_t num_features = X.cols();
        VecType eps(num_components_);
        eps.fill(10.0 * ConstType<DataType>::min());

        VecType nk(num_components_);
        nk = math::sum<MatType, VecType>(resp, 0);
        nk = nk.array() + eps.array(); 

        MatType nk_tmp;
        nk_tmp = utils::repeat<MatType>(nk, num_features, 1);
        
        MatType means_tmp = resp.transpose() * X;
        MatType means = means_tmp.array() / nk_tmp.array();

        std::vector<MatType> covariances;

        if (cov_type == "full") {
            covariances = estimate_cov_full(X, resp, means, nk, reg_covar);
        }
        
        return std::make_tuple(covariances, means, nk);
    }

    /**
     * 
    */
    void initialize_parameters(const MatType& X) {
        std::size_t num_samples = X.rows();
        MatType resp(num_samples, num_components_);
        if (init_params_ == "random") {
            resp = math::rand<MatType>(num_samples, num_components_);

            VecType sum_resp = math::sum<MatType, VecType>(resp, 1);
            MatType sum_resp_tmp = utils::repeat<MatType>(sum_resp, num_components_, 1);
            resp = resp.array() / sum_resp_tmp.array();
        }

        // std::cout << resp << std::endl;
        std::vector<MatType> covariances;
        MatType means;
        VecType weights;
        std::tie(covariances, 
            means, 
            weights) = estimate_gaussian_parameters(X, resp, reg_covar_);

        weights = weights * (static_cast<DataType>(1) / static_cast<DataType>(num_samples));


        precisions_cholesky_ = compute_precision_cholesky(covariances);


        // for(auto& cov : covariances) {
        //     std::cout << cov << std::endl;
        // }
        // std::cout << means << std::endl;
        // std::cout << weights << std::endl;
        // std::cout << "--------------------------------" << std::endl;

        // for(auto& precision : precisions_cholesky_) {

        //     std::cout << precision << std::endl;
        //     std::cout << precision.diagonal() << std::endl;
        // }

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

        initialize_parameters(X);

        std::size_t num_features = X.cols();

        VecType log_det_chol = compute_log_det_cholesky(precisions_cholesky_, num_features);

        std::cout << log_det_chol << std::endl;

    }












};

} // mixture_model
} // openml

#endif /*METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP*/
