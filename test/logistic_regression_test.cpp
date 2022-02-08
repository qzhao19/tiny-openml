#include "../src/methods/linear_model/logistic_regression.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);

    MatType X_train;
    MatType X_test;
    VecType y_train;
    VecType y_test;
    std::tie(X_train, X_test, y_train, y_test) = data::train_test_split<MatType, VecType>(X, y, 0.9);

    linear_model::LogisticRegression<double> lr;   
    lr.fit(X_train, y_train);

    VecType y_pred;
    y_pred = lr.predict(X_test);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;

    MatType y_pred_prob;
    y_pred_prob = lr.predict_prob(X_test);
    std::cout << "y_pred_prob" << std::endl;
    std::cout << y_pred_prob << std::endl;



}
