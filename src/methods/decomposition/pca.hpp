#ifndef METHODS_DECOMPOSITION_PCA_HPP
#define METHODS_DECOMPOSITION_PCA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "./base.hpp"
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
class PCA: public BaseDecompositionModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::string solver_;
    double noise_var_;

protected:
    void fit_data(const MatType& X) {
        std::size_t num_samples = X.rows(), num_features = X.cols();

        MatType centered_X;
        centered_X = math::center<MatType>(X);
        
        MatType components;
        VecType explained_var;
        VecType explained_var_ratio;

        VecType s;
        MatType U, Vt;
        if (solver_ == "svd") {
            std::tie(U, s, Vt) = math::exact_svd<MatType, VecType>(centered_X, false);
            // flip eigenvectors' sign to enforce deterministic output
            Vt = Vt.transpose().eval();
            std::tie(U, Vt) = math::svd_flip<MatType, VecType, IdxVecType>(U, Vt);

            explained_var = math::power<VecType>(s, 2.0) / (static_cast<DataType>(num_samples) - 1.);
            components = Vt;
        };

        VecType total_var = math::sum<MatType>(explained_var);
        explained_var_ratio = explained_var / total_var.value();

        if (this->num_components_ < std::min(num_samples, num_features)) {
            std::size_t num_noise_var = std::min(num_samples, num_features) - this->num_components_;
            auto noise_var = math::mean<MatType>(explained_var.bottomRows(num_noise_var));
            noise_var_ = static_cast<double>(noise_var.value());
        }
        else {
            noise_var_ = 0.0;
        }
        this->singular_values_ = s.topRows(this->num_components_);
        this->components_ = components.topRows(this->num_components_);
        this->explained_var_ = explained_var.topRows(this->num_components_);
        this->explained_var_ratio_ = explained_var_ratio.topRows(this->num_components_);
    }

    /** transform the new data */
    const MatType transform_data(const MatType& X) const{
        std::size_t num_samples = X.rows();
        MatType centered_X;
        centered_X = math::center<MatType>(X);

        MatType transformed_X(num_samples, this->num_components_);
        transformed_X = centered_X * this->components_.transpose();
        return transformed_X; 
    }

    /**calculate the covariance*/
    void compute_data_covariance() {
        std::size_t num_features = this->components_.cols();
        // MatType components_ = components;

        MatType explained_var(this->num_components_, this->num_components_);
        explained_var = math::diagmat<MatType, VecType>(this->explained_var_);

        MatType explained_var_diff = explained_var.array() - noise_var_;
        explained_var_diff = explained_var_diff.cwiseMax(static_cast<DataType>(0));

        MatType cov(num_features, num_features);
        cov = this->components_.transpose() * explained_var_diff * this->components_; 

        MatType eye(num_features, num_features);
        eye.setIdentity();
        cov += (eye * noise_var_).eval();

        this->covariance_ = cov;
    }

    /**Compute data precision matrix*/
    void compute_precision_matrix() {
        this->precision_ = math::pinv<MatType, VecType>(this->covariance_);
    }

    /**compute log_likelihood of each sample*/
    const MatType compute_score_samples(const MatType& X) const {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        MatType log_like_(num_samples, num_features);
        log_like_ = X.array() * (X * this->precision_).array();

        VecType log_like(num_samples);
        log_like = math::sum<MatType>(log_like_, 1);
        log_like = log_like * (-0.5);

        log_like.array() -= 0.5 * (static_cast<DataType>(num_features) * \
            std::log(2.0 * M_PI) - \
                math::logdet<MatType>(this->precision_));

        return log_like;
    }

public:
    PCA(): BaseDecompositionModel<DataType>(2), 
        solver_("svd") {};

    PCA(std::string solver, 
        std::size_t num_components): 
            BaseDecompositionModel<DataType>(num_components),
            solver_(solver) {};

    /**deconstructor*/
    ~PCA() {};

    /**
     * Return the average log-likelihood of all samples.
     * @param X array-like of shape (n_samples, n_features)
     *      the data
     * @return Average log-likelihood of the samples 
     *      under the current model.
    */
    const double score(const MatType& X) const {
        std::size_t num_samples = X.rows();
        
        MatType centered_X;
        centered_X = math::center<MatType>(X);

        VecType log_like(num_samples);
        log_like = compute_score_samples(centered_X);

        auto average_log_like = math::mean<MatType>(log_like, -1);

        return average_log_like.value();
    }

     /**
     * Compute data covariance with the generative model.
     * ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
     * where S**2 contains the explained variances, and sigma2 contains the noise variances.
     * 
     * @return covariance : array of shape (n_features, n_features)
     * Estimated covariance of data. 
    */
    const MatType get_covariance() {
        compute_data_covariance();
        return this->covariance_;
    }

    /**
     * Compute data precision matrix, 
     * equals the inverse of the covariance
     * 
    */
    const MatType get_precision() {
        compute_precision_matrix();
        return this->precision_;
    }

};

} // namespace openml
} // namespace decomposition

#endif /*METHODS_DECOMPOSITION_PCA_HPP*/
