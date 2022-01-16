#include "../src/methods/linear_model/linear_regression.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/boston_house_price.txt", X, y);

    MatType X_train;
    MatType X_test;
    VecType y_train;
    VecType y_test;
    std::tie(X_train, X_test, y_train, y_test) = data::train_test_split<MatType, VecType>(X, y, 0.9);

    linear_model::LinearRegression<double> lr;   
    lr.fit(X_train, y_train);

    VecType y_pred;
    y_pred = lr.predict(X_test);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;

    MatType weights;
    weights = lr.get_coef();
    std::cout << "weights" << std::endl;
    std::cout << weights << std::endl;

}
