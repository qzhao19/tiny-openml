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
    std::size_t num_init_;
    std::size_t num_components_;
    std::string init_params_;
    std::string covariance_type_;
    double tol_;
    double reg_covar_;
    

    std::vector<MatType> precisions_cholesky_;
    std::vector<MatType> covariances_;
    MatType means_;
    VecType weights_;
    double lower_bound_;

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
        const std::vector<MatType>& matrix_chol) const {
        
        VecType log_det_chol(num_components_);
        for (std::size_t i = 0; i < num_components_; ++i) {
            VecType diag = matrix_chol[i].diagonal();
            auto tmp = math::sum<MatType, VecType>(diag.array().log(), 0);
            log_det_chol(i) = tmp.value();
        }

        return log_det_chol;
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
     * Estimate the log Gaussian probability.
     * The determinant of the precision matrix from the Cholesky decomposition
     * corresponds to the negative half of the determinant of the full precision
     * matrix.
     * 
    */
    const MatType estimate_log_prob(
        const std::vector<MatType>& precision_chol,
        const MatType& X, 
        const MatType& means) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_components = means.rows();

        VecType log_det = compute_log_det_cholesky(precisions_cholesky_);

        MatType log_prob(num_samples, num_components);

        for (std::size_t i = 0; i < num_components; ++i) {
            MatType prec_chol = precision_chol[i];
            VecType mu = means.row(i).transpose();

            MatType tmp1(num_samples, num_features);
            tmp1 = X * prec_chol;

            MatType tmp2(1, num_features);
            tmp2 = mu.transpose() * prec_chol;

            MatType tmp3(num_samples, num_features);
            tmp3 = utils::repeat<MatType>(tmp2, num_samples, 0);

            MatType y = tmp1.array() - tmp3.array();
            MatType y_square = y.array().square();

            log_prob.col(i) = math::sum<MatType, VecType>(y_square, 1);

        }

        DataType v = static_cast<DataType>(num_features) * 
            static_cast<DataType>(std::log(2 * M_PI)) * 
                static_cast<DataType>(-0.5);
        MatType C(num_samples, num_components);
        C.fill(v);

        MatType tmp4(num_samples, num_components);
        tmp4 = utils::repeat<MatType>(log_det.transpose(), num_samples, 0);

        return log_prob * (-0.5) + C + tmp4;
    }


    /**
     * Estimate the weighted log-probabilities, 
     * log P(X|Z) + log_weights
    */
    const MatType estimate_weighted_log_prob(
        const MatType& X) const {
        
        std::size_t num_samples = X.rows();

        MatType log_prob;
        log_prob = estimate_log_prob(precisions_cholesky_, X, means_);

        VecType log_weights;
        log_weights = weights_.array().log();

        MatType tmp(num_samples, num_components_);
        tmp = utils::repeat<MatType>(log_weights.transpose(), num_samples, 0);

        MatType weighted_log_prob;
        weighted_log_prob = log_prob + tmp;

        return weighted_log_prob;

    }


    /**
     * Estimate log prob and resp for each sample.
     * Compute the log prob, weighted log prob per
     * component and resp for each sample in X 
    */
    const std::tuple<MatType, VecType> 
    estimate_log_prob_resp(
        const MatType& X) const {
        
        std::size_t num_samples = X.rows();

        MatType weighted_log_prob(num_samples, num_components_);
        weighted_log_prob = estimate_weighted_log_prob(X);

        VecType log_prob_norm(num_samples);
        log_prob_norm = math::logsumexp<MatType, VecType>(weighted_log_prob, 1);

        MatType tmp(num_samples, num_components_);
        tmp = utils::repeat<MatType>(log_prob_norm, num_components_, 1);

        MatType log_resp(num_samples, num_components_);
        log_resp = weighted_log_prob - tmp;

        return std::make_tuple(log_resp, log_prob_norm);

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
     * Initialize the model parameters.
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

        std::vector<MatType> covariances;
        MatType means;
        VecType weights;
        std::tie(covariances, 
            means, 
            weights) = estimate_gaussian_parameters(X, resp, reg_covar_);
        
        means_ = means;
        weights = weights * (static_cast<DataType>(1.0) / static_cast<DataType>(num_samples));
        weights_ = weights;

        covariances_ = covariances;

        precisions_cholesky_ = compute_precision_cholesky(covariances);

    }

    /**
     * E step.
     * @param X ndarray of shape [num_samples, num_features]
     * @return log_prob_norm: numbers
     *      The mean of log of prob of each samples
     * @return log_resp: ndarray of shape [num_samples, num_components]
     *      log of the post prob of the each samples in X
    */
    std::tuple<DataType, MatType> e_step(
        const MatType& X) const {
        
        MatType log_resp;
        VecType log_prob_norm;
        std::tie(log_resp, log_prob_norm) = estimate_log_prob_resp(X);

        VecType mean_log_prob;
        mean_log_prob = math::mean<MatType, VecType>(log_prob_norm, 0);

        return std::make_tuple(mean_log_prob.value(), log_resp);
    }


    /**
     * M step.
     * 
    */
    void m_step(const MatType& X, const MatType& log_resp) {
        std::size_t num_samples = X.rows();
        MatType resp = log_resp.array().exp();

        std::vector<MatType> covariances;
        MatType means;
        VecType weights;
        std::tie(covariances, 
            means, 
            weights) = estimate_gaussian_parameters(X, resp, reg_covar_);

        means_ = means;
        weights = weights * (static_cast<DataType>(1.0) / static_cast<DataType>(num_samples));
        weights_ = weights;

        covariances_ = covariances;

        precisions_cholesky_ = compute_precision_cholesky(covariances);
    }



    /**
     * Estimate model parameters using X
     * std::ostringstream err;
     * err << "Unable to open output file: " << szOutFilePath << std::endl;
     * throw std::invalid_argument(err.str());
    */
    void fit_data(const MatType& X){

        std::size_t num_samples = X.rows(), num_features = X.cols();
        if (num_samples < num_components_) {
            char buffer[200];
            std::snprintf(buffer, 200, 
                "Expected n_samples >= n_components, but got num_components = [%ld], num_samples = [%ld]", 
                num_components_, num_samples);
            std::string err_msg = static_cast<std::string>(buffer);
            throw std::out_of_range(err_msg);
        }

        bool converged = false;
        double max_lower_bound = ConstType<double>::min();
        

        std::vector<MatType> best_precisions_cholesky;
        std::vector<MatType> best_covariances;
        MatType best_means;
        VecType best_weights;


        for (std::size_t init = 0; init < num_init_; ++init) {

            initialize_parameters(X);
            double lower_bound = ConstType<double>::min();
            for (std::size_t iter = 0; iter < max_iter_; ++iter) {
                double prev_lower_bound = lower_bound;

                DataType mean_log_prob;
                MatType log_resp;
                std::tie(mean_log_prob, log_resp) = e_step(X);
                m_step(X, log_resp);

                lower_bound = static_cast<double>(mean_log_prob);
                double diff = lower_bound - prev_lower_bound;
                if (std::abs(diff) < tol_) {
                    converged = true;
                    break;
                }
            }

            if ((lower_bound > max_lower_bound) || 
                (max_lower_bound == ConstType<double>::min())) {
                
                max_lower_bound = lower_bound;
                best_precisions_cholesky = precisions_cholesky_;
                best_covariances = covariances_;
                best_means = means_;
                best_weights = weights_;
            }
        }

        if (!converged) {
            std::cout << "Not converge, try different init parameters." << std::endl;
        }

        lower_bound_ = max_lower_bound;
        precisions_cholesky_ = best_precisions_cholesky;
        covariances_ = best_covariances;
        means_ = best_means;
        weights_ = best_weights;

        DataType mean_log_prob;
        MatType log_resp;
        std::tie(mean_log_prob, log_resp) = e_step(X);
    }




public:
    GaussianMixture(): max_iter_(100), 
        num_init_(1),
        num_components_(2), 
        init_params_("random"), 
        covariance_type_("full"), 
        tol_(1e-3), 
        reg_covar_(1e-6) {};

    GaussianMixture(std::size_t max_iter,
        std::size_t num_init,
        std::size_t num_components,
        std::string init_params,
        std::string covariance_type,
        double tol,
        double reg_covar): max_iter_(max_iter), 
            num_init_(num_init),
            num_components_(num_components), 
            init_params_(init_params), 
            covariance_type_(covariance_type), 
            tol_(tol), 
            reg_covar_(reg_covar) {};



    void test_func(const MatType& X) {

        initialize_parameters(X);

        std::size_t num_features = X.cols();

        VecType log_det_chol = compute_log_det_cholesky(precisions_cholesky_);

        // std::cout << "log_det_chol" << std::endl;
        // std::cout << log_det_chol << std::endl;

        MatType log_prob = estimate_log_prob(precisions_cholesky_, X, means_);

        MatType log_resp;
        VecType log_prob_norm;
        std::tie(log_resp, log_prob_norm) = estimate_log_prob_resp(X);

        // std::cout << math::mean<MatType, VecType>(log_prob_norm, 0) << std::endl;

        // std::cout << "log_prob" << std::endl;
        // std::cout << log_prob << std::endl;

        // std::cout << "log_resp" << std::endl;
        // std::cout << log_resp << std::endl;

        // std::cout << "log_prob_norm" << std::endl;
        // std::cout << log_prob_norm << std::endl;

        // for(auto& cov : covariances_) {
        //     std::cout << "cov" << std::endl;
        //     std::cout << cov << std::endl;
        // }
        // std::cout << "means" << std::endl;
        // std::cout << means_ << std::endl;
        // std::cout << "weights" << std::endl;
        // std::cout << weights_ << std::endl;
        // std::cout << "--------------------------------" << std::endl;
        // for(auto& precision : precisions_cholesky_) {
        //     std::cout << "precision" << std::endl;
        //     std::cout << precision << std::endl;
        // }
    }



};

} // mixture_model
} // openml

#endif /*METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP*/
