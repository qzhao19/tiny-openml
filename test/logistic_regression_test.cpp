#include "../src/methods/linear_model/logistic_regression.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);

    std::size_t num_features = X.cols();
    std::cout << "sgd test" <<std::endl;
    
    VecType w(num_features);
    w.setOnes();
    VecType opt_w(num_features);

    linear_model::LogisticRegression<double> lr;   
    lr.fit(X, y);

    VecType y_pred;
    y_pred = lr.predict(X);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;

    MatType y_pred_prob;
    y_pred_prob = lr.predict_prob(X);
    std::cout << "y_pred_prob" << std::endl;
    std::cout << y_pred_prob << std::endl;

}
