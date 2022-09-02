#include "../src/methods/naive_bayes/gaussian_naive_bayes.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    naive_bayes::GaussianNB<double> gaussian_nb;

    gaussian_nb.fit(X, y);

    MatType log_prob = gaussian_nb.predict_log_prob(X);
    std::cout << "log_prob" << std::endl;
    std::cout << log_prob << std::endl;

    MatType prob = gaussian_nb.predict_prob(X);
    std::cout << "prob" << std::endl;
    std::cout << prob << std::endl;

    VecType y_pred = gaussian_nb.predict(X);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;

    std::cout << "prior prob" << std::endl;
    std::cout << gaussian_nb.get_prior_prob() << std::endl;

    std::cout << "classes" << std::endl;
    std::cout << gaussian_nb.get_classes() << std::endl;


}