#include "../src/methods/mixture/gaussian_mixture.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    mixture_model::GaussianMixture<double> gmm;

    gmm.fit(X);
    // gmm.test_func(X); 
    // std::cout << "--------------------------------" << std::endl;

    VecType y_pred;
    y_pred = gmm.predict(X);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;
    std::cout << "--------------------------------" << std::endl;

    MatType pred_prob;
    pred_prob = gmm.predict_prob(X);
    std::cout << "y_pred" << std::endl;
    std::cout << pred_prob << std::endl;
    std::cout << "--------------------------------" << std::endl;

    std::vector<MatType> covariances;
    std::vector<MatType> precisions;
    MatType means;
    VecType weights;

    covariances = gmm.get_covariance();
    precisions = gmm.get_precisions();
    means = gmm.get_means();
    weights = gmm.get_weights();


    for(auto& cov : covariances) {
            std::cout << "cov" << std::endl;
            std::cout << cov << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
    for(auto& prec : precisions) {
        std::cout << "precision" << std::endl;
        std::cout << prec << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
    std::cout << "means" << std::endl;
    std::cout << means << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "weights" << std::endl;
    std::cout << weights << std::endl;
    

}