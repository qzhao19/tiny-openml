#include "../src/methods/linear_model/softmax_regression.hpp"
using namespace openml;


int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;
    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    std::cout << "Softmax regression using sgd optimizer" <<std::endl;
    linear_model::SoftmaxRegression<double> regressor(
        0.01, 0.0, 0.0001, 16, 2000, 5, "sgd", "None", true, false
    );
    regressor.fit(X, y);

    VecType y_pred;
    y_pred = regressor.predict(X);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;

    MatType y_pred_prob;
    y_pred_prob = regressor.predict_prob(X);
    std::cout << "y_pred_prob" << std::endl;
    std::cout << y_pred_prob << std::endl;

};
