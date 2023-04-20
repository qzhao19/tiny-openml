#ifndef METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP
#define METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace mixture_model {

/**
 * Gaussian Mixture Model 
 * @param max_iter: int default 1
 *    The number of EM iterations to perform.
 * @param num_init: int default 1
 *    The number of initializations to perform
 * @param num_components: int default 1
 *    The number of mixture components.
 * @param init_params: string, default random
 *    The method used to initialize the weights, the means and the precisions
 *    'random' : resp are initialized randomly.
 * @param covariance_type: string of {full, tied}, default 'full'
 *    The type of covariance parameters to use
 *    1. ‘full’: each component has its own general covariance matrix.
 *    2. ‘tied’: all components share the same general covariance matrix.
 * @param tol: double, default 0.003
 *    The convergence threshold.
 * @param reg_covar_: double, default 1e-6
 *    Non-negative regularization added to the diagonal of covariance
*/
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
    
    std::vector<MatType> covariances_;
    std::vector<MatType> precisions_;
    std::vector<MatType> precisions_cholesky_;
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
        for (std::size_t i = 0; i < covariances.size(); ++i) {
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
     * @param matrix_chol : array-like of [n_components, n_features, n_features]
     *      Cholesky decompositions of the matrices.
     * @return log_det_precision_chol, The determinant of the precision 
     *      matrix for each component
    */
    const VecType compute_log_det_cholesky(
        const std::vector<MatType>& matrix_chol) const {
        
        std::size_t num_chol = matrix_chol.size();

        VecType log_det_chol(num_chol);
        for (std::size_t i = 0; i < num_chol; ++i) {
            VecType diag = matrix_chol[i].diagonal();
            auto tmp = math::sum<MatType>(diag.array().log(), 0);
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
    const std::vector<MatType> estimate_gaussian_covariances_full(const MatType& X, 
        const MatType& resp, 
        const MatType& means, 
        const VecType& nk, 
        double reg_covar) const {
        
        std::size_t num_samples = X.rows();
        std::size_t num_components = means.rows(), num_features = means.cols();
        std::vector<MatType> covariances;

        VecType reg(num_features);
        reg.fill(reg_covar);

        for (std::size_t k = 0; k < num_components; k++) {
            MatType diff(num_samples, num_features);
            for (std::size_t i = 0; i < num_samples; ++i) {
                diff.row(i) = X.row(i).array() - means.row(k).array();
            }

            MatType resp_tmp = common::repeat<MatType>(resp.col(k), num_features, 1);
            // matrix product dot 
            MatType resp_diff_tmp = resp_tmp.array() * diff.array();
            MatType resp_diff = resp_diff_tmp.transpose() * diff;
            
            MatType nk_tmp(num_features, num_features);
            DataType val = nk(k);
            nk_tmp.fill(val);

            // compute covariance for each component
            MatType cov = resp_diff.array() / nk_tmp.array(); 
            cov = cov + MatType(reg.asDiagonal());
            covariances.push_back(cov);
        }
        return covariances;
    }


    /**
     * Estimate the tied covariance matrix.
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
    const std::vector<MatType> estimate_gaussian_covariances_tied(const MatType& X, 
        const MatType& resp, 
        const MatType& means, 
        const VecType& nk, 
        double reg_covar) const {
        
        std::size_t num_samples = X.rows();
        std::size_t num_components = means.rows(), num_features = means.cols();
        std::vector<MatType> covariances;

        VecType reg(num_features);
        reg.fill(reg_covar);

        MatType avg_X = X.transpose() * X;
        // MatType avg_means = 

        MatType nk_tmp(num_components, num_features);
        nk_tmp = common::repeat<MatType>(nk, num_features, 1);
        auto sum_nk = math::sum<MatType>(nk, 0);

        MatType avg_means = (nk_tmp.array() * means.array()).matrix().transpose() * means;

        MatType cov = avg_X - avg_means;
        cov = cov.array() / sum_nk.value();
        cov = cov + MatType(reg.asDiagonal());

        covariances.push_back(cov);

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
        std::string covariance_type) {
        
        std::size_t num_features = X.cols();
        VecType eps(num_components_);
        eps.fill(10.0 * ConstType<DataType>::epsilon());

        VecType nk(num_components_);
        nk = resp.colwise().sum().transpose();
        nk = nk.array() + eps.array(); 

        MatType nk_tmp;
        nk_tmp = common::repeat<MatType>(nk, num_features, 1);
        
        MatType means_tmp = resp.transpose() * X;
        MatType means = means_tmp.array() / nk_tmp.array();

        std::vector<MatType> covariances;

        if (covariance_type == "full") {
            covariances = estimate_gaussian_covariances_full(X, resp, means, nk, reg_covar);
        }
        else if (covariance_type == "tied") {
            covariances = estimate_gaussian_covariances_tied(X, resp, means, nk, reg_covar);
        }
        
        return std::make_tuple(covariances, means, nk);
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
            MatType prec_chol;
            if (covariance_type_ == "tied") {
                prec_chol = precision_chol[0];
            }
            else if (covariance_type_ == "full") {
                prec_chol = precision_chol[i];
            }
            VecType mu = means.row(i).transpose();

            MatType tmp1(num_samples, num_features);
            tmp1 = X * prec_chol;

            MatType tmp2(1, num_features);
            tmp2 = mu.transpose() * prec_chol;

            MatType tmp3(num_samples, num_features);
            tmp3 = common::repeat<MatType>(tmp2, num_samples, 0);

            MatType y = tmp1.array() - tmp3.array();
            MatType y_square = y.array().square();

            log_prob.col(i) = math::sum<MatType>(y_square, 1);
        }

        DataType v = static_cast<DataType>(num_features) * 
            static_cast<DataType>(std::log(2 * M_PI)) * 
                static_cast<DataType>(-0.5);
        MatType C(num_samples, num_components);
        C.fill(v);

        MatType tmp4(num_samples, num_components);
        tmp4 = common::repeat<MatType>(log_det.transpose(), num_samples, 0);

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
        tmp = common::repeat<MatType>(log_weights.transpose(), num_samples, 0);

        MatType weighted_log_prob;
        weighted_log_prob = log_prob + tmp;

        return weighted_log_prob;
    }


    /**
     * Estimate log prob and resp for each sample.
     * Compute the log prob, weighted log prob per
     * component and resp for each sample in X 
    */
    const std::tuple<MatType, VecType> estimate_log_prob_resp(
        const MatType& X) const {
        
        std::size_t num_samples = X.rows();

        MatType weighted_log_prob(num_samples, num_components_);
        weighted_log_prob = estimate_weighted_log_prob(X);

        VecType log_prob_norm(num_samples);
        log_prob_norm = math::logsumexp<MatType, VecType>(weighted_log_prob, 1);

        MatType tmp(num_samples, num_components_);
        tmp = common::repeat<MatType>(log_prob_norm, num_components_, 1);

        MatType log_resp(num_samples, num_components_);
        log_resp = weighted_log_prob - tmp;

        return std::make_tuple(log_resp, log_prob_norm);

    }

    /**
     * Initialize the model parameters.
    */
    void initialize_parameters(const MatType& X) {
        std::size_t num_samples = X.rows();
        MatType resp(num_samples, num_components_);
        if (init_params_ == "random") {
            resp = random::rand<MatType>(num_samples, num_components_);

            VecType sum_resp = math::sum<MatType>(resp, 1);
            MatType sum_resp_tmp = common::repeat<MatType>(sum_resp, num_components_, 1);
            resp = resp.array() / sum_resp_tmp.array();
        }

        std::vector<MatType> covariances;
        MatType means;
        VecType weights;
        std::tie(covariances, 
            means, 
            weights) = estimate_gaussian_parameters(X, resp, reg_covar_, covariance_type_);
        
        means_ = means;
        weights = weights / static_cast<DataType>(num_samples);
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

        // VecType mean_log_prob;
        auto mean_log_prob = math::mean<MatType>(log_prob_norm, 0);

        return std::make_tuple(mean_log_prob.value(), log_resp);
    }

    /**
     * M step.
     * @param X ndarray of shape [num_samples, num_features]
     * @param log_resp: ndarray of shape [num_samples, num_components]
     *    log of the post prob of the each samples in X
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
            weights) = estimate_gaussian_parameters(X, resp, reg_covar_, covariance_type_);

        means_ = means;
        weights = weights / static_cast<DataType>(num_samples);
        weights_ = weights;

        covariances_ = covariances;
        precisions_cholesky_ = compute_precision_cholesky(covariances_);

    }

    /**
     * Estimate model parameters using X
    */
    void fit_data(const MatType& X) {

        std::size_t num_samples = X.rows(), num_features = X.cols();
        if (num_samples < num_components_) {
            std::ostringstream err_msg;
            err_msg << "Expected n_samples >= n_components, "
                    << "but got num_components = " << num_components_
                    << ", num_samples = " << num_samples << std::endl;
            throw std::out_of_range(err_msg.str());
        }

        bool converged = false;
        std::size_t best_num_iters;
        double max_lower_bound = -ConstType<double>::infinity();
        
        std::vector<MatType> best_precisions_cholesky;
        std::vector<MatType> best_covariances;
        MatType best_means;
        VecType best_weights;

        for (std::size_t init = 0; init < num_init_; ++init) {

            initialize_parameters(X);
            double lower_bound = -ConstType<double>::infinity();
            std::size_t iter;
            for (iter = 0; iter < max_iter_; ++iter) {
                double prev_lower_bound = lower_bound;

                DataType log_prob_norm;
                MatType log_resp;
                std::tie(log_prob_norm, log_resp) = e_step(X);
                m_step(X, log_resp);

                lower_bound = static_cast<double>(log_prob_norm);
                double diff = lower_bound - prev_lower_bound;

                if (std::abs(diff) < tol_) {
                    converged = true;
                    break;
                }
            }

            if ((lower_bound > max_lower_bound) || 
                (max_lower_bound == -ConstType<double>::infinity())) {
                
                max_lower_bound = lower_bound;
                best_precisions_cholesky = precisions_cholesky_;
                best_covariances = covariances_;
                best_means = means_;
                best_weights = weights_;
                best_num_iters = iter;
            }
        }

        if (!converged) {
            throw std::runtime_error("Not converge, try different init parameters.");
        }

        lower_bound_ = max_lower_bound;
        precisions_cholesky_ = best_precisions_cholesky;
        covariances_ = best_covariances;
        means_ = best_means;
        weights_ = best_weights;

        for (int i = 0; i < precisions_cholesky_.size(); ++i) {
            MatType prec_chol = precisions_cholesky_[i];
            MatType prec(num_features, num_features);
            prec = prec_chol * prec_chol.transpose();
            precisions_.push_back(prec);
        }

        DataType mean_log_prob;
        MatType log_resp;
        std::tie(mean_log_prob, log_resp) = e_step(X);
    }

public:
    GaussianMixture(): max_iter_(100), 
        num_init_(1),
        num_components_(3), 
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

    ~GaussianMixture() {};

    /**
     * fit dataset with EM algorithm
    */
    void fit(const MatType& X) {
        fit_data(X);
    }

    /**
     * Predict the labels for the input dataset
     * @param X  ndarray of shape [num_samples, num_features]
     *    the input dataset
    */
    VecType predict(const MatType& X) {
        std::size_t num_samples = X.rows();

        MatType weighted_log_prob(num_samples, num_components_);
        weighted_log_prob = estimate_weighted_log_prob(X);

        auto argmax_index = common::argmax<MatType, VecType, IdxType>(weighted_log_prob, 1);

        VecType y_pred = argmax_index.template cast<DataType>();
        return y_pred;
    }

    /**
     * Evaluate the components' proba for each sample.
    */
    MatType predict_prob(const MatType& X) {

        MatType log_resp;
        VecType log_prob_norm;
        std::tie(log_resp, log_prob_norm) = estimate_log_prob_resp(X);

        return log_resp.array().exp();
    }


    /**
     * get the covariance of each mixture component.
    */
    const std::vector<MatType> get_covariance() const {
        return covariances_;
    }

    /**
     * get precision matrices for each component in the mixture
     * 
     * "A precision matrix is the inverse of a covariance matrix. 
     * A covariance matrix is symmetric positive definite so the 
     * mixture of Gaussian can be equivalently parameterized by 
     * the precision matrices."  --sklearn
    */
    const std::vector<MatType> get_precisions() const {
        return precisions_;
    }

    /**
     * The cholesky decomposition of the precision matrices
    */
    const std::vector<MatType> get_precisions_cholesky() const {
        return precisions_cholesky_;
    }

    /**
     * get the mean (ndarray) of each mixture component
    */
    const MatType get_means() const {
        return means_;
    }

    /**
     * get the weights of each mixture components.
    */
    const VecType get_weights() const {
        return weights_;
    }

    /**
     * Lower bound value on the log-likelihood
    */
    const double get_lower_bound() const {
        return lower_bound_;
    }

};

} // mixture_model
} // openml

#endif /*METHODS_MIXTURE_GAUSSIAN_MIXTURE_HPP*/
