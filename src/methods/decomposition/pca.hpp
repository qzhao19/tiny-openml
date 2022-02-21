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
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    /**
     * @param components: ndarray of shape (n_components, n_features)
     *      Principal axes in feature space, representing the directions of 
     *      maximum variance in the data.
     * @param explained_var: ndarray of shape (n_components,)
     *      The amount of variance explained by each of the selected components. 
     * @param explained_variance_ratio: ndarray of shape (n_components,)
     *      Percentage of variance explained by each of the selected components.
    */
    MatType precision;
    MatType covariance;
    MatType components;
    VecType explained_var;
    VecType explained_var_ratio;

    std::string solver;
    std::size_t num_components;
    double noise_var;

protected:

    void fit_data(const MatType& X) {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        MatType components_;
        VecType explained_var_;
        VecType explained_var_ratio_;

        if (solver == "svd") {
            MatType U;
            VecType s; 
            MatType V;
            std::tie(U, s, V) = math::exact_svd<MatType, VecType>(X, false);
            
            // flip eigenvectors' sign to enforce deterministic output
            MatType Vt = V.transpose();
            std::tie(U, Vt) = math::svd_flip<MatType, VecType, IdxType>(U, Vt);

            explained_var_ = math::power<VecType>(s, 2.0) / (static_cast<DataType>(num_samples) - 1.);
            components_ = Vt;
        };

        VecType total_var = math::sum<MatType, VecType>(explained_var_);
        explained_var_ratio_ = explained_var_ / total_var(0, 0);

        if (num_components < std::min(num_samples, num_features)) {
            std::size_t num_noise_var = std::min(num_samples, num_features) - num_components;
            VecType noise_var_ = math::mean<MatType, VecType>(explained_var_.bottomRows(num_noise_var));
            noise_var = static_cast<double>(noise_var_(0, 0));
        }
        else {
            noise_var = 0.0;
        }
        components = components_.topRows(num_components);
        explained_var = explained_var_.topRows(num_components);
        explained_var_ratio = explained_var_ratio_.topRows(num_components);

    }

    /** transform the new data */
    const MatType transform_data(const MatType& X) const{
        std::size_t num_samples = X.rows();
        
        MatType transformed_X(num_samples, num_components);
        transformed_X = X * components.transpose();
        return transformed_X; 
    }

    /**calculate the covariance*/
    void compute_data_covariance() {
        std::size_t num_features = components.cols();
        // MatType components_ = components;

        MatType explained_var_(num_components, num_components);
        explained_var_ = math::diagmat<MatType, VecType>(explained_var);

        MatType explained_var_diff = explained_var_.array() - noise_var;
        explained_var_diff = explained_var_diff.matrix().cwiseMax(static_cast<DataType>(0));

        MatType cov(num_features, num_features);
        cov = components.transpose() * explained_var_diff * components; 

        MatType eye(num_features, num_features);
        eye.setIdentity();
        cov += eye * noise_var;

        covariance = cov;
    }

    /**Compute data precision matrix*/
    void compute_precision_matrix() {
        precision = math::pinv<MatType, VecType>(covariance);
    }

    /**compute log_likelihood of each sample*/
    const MatType compute_score_samples(const MatType& X) const {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        MatType log_like_(num_samples, num_features);
        log_like_ = (X.array() * (X * precision).array()).matrix();

        VecType log_like(num_samples);
        log_like = math::sum<MatType, VecType>(log_like_, 1);
        log_like = log_like * (-0.5);

        log_like.array() -= 0.5 * (static_cast<DataType>(num_features) * \
            std::log(2.0 * M_PI) - \
                math::logdet<MatType>(precision));

        return log_like.matrix();
    }

public:
    /**
     * Default constructor to create PCA object, linear dimesionality reduction using SVD 
     * of the data to project it to a lower dimesional space, input data shoule be centered 
     * 
     * @param solver the matrix decomnposition policies, 
     *      if svd, will run full svd via math::exact_svd
     * 
     * @param n_components Number of components to keep
     * @param scale Whether or not to scale the data.
    */
    PCA(): solver("svd"), 
        num_components(2) {};


    PCA(std::string solver_, 
        std::size_t num_components_): 
            solver(solver_), 
            num_components(num_components_) {};

    /**deconstructor*/
    ~PCA() {};

    /**
     * Fit the model with X.
     * @param X array-like of shape (num_samples, num_features)
     *      Training data, where num_samples is the number of 
     *      samples and num_features is the number of features.
    */
    void fit(const MatType& X) {
        MatType centered_X;
        centered_X = math::center<MatType>(X);
        fit_data(centered_X);
    }

    /**
     * Apply dimensionality reduction to X. X is projected on the 
     * first principal components previously extracted from a 
     * training set.
     * 
     * @param X array-like of shape (num_samples, num_features)
     *      New data
     * @return X_new array-like of shape
     *      Projection of X in the first principal components
    */
    const MatType transform(const MatType& X) const{
        MatType centered_X;
        centered_X = math::center<MatType>(X);

        MatType transformed_X;
        transformed_X = transform_data(centered_X); 
        return transformed_X;
    }

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

        auto average_log_like = math::mean<MatType, VecType>(log_like);

        return average_log_like(0, 0);
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
        return covariance;
    }

    /**
     * Compute data precision matrix, 
     * equals the inverse of the covariance
     * 
    */
    const MatType get_precision() {
        compute_precision_matrix();
        return precision;
    }

    /**override get_coef interface*/
    const VecType get_components() const {
        return components;
    };

    const VecType get_explained_var() const {
        return explained_var;
    };

    const VecType get_explained_var_ratio() const {
        return explained_var_ratio;
    };

};

} // namespace openml
} // namespace regression

#endif /*METHODS_DECOMPOSITION_PCA_HPP*/
