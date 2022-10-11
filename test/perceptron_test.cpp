#include "../src/methods/linear_model/perceptron.hpp"
using namespace openml;


int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);


    std::size_t num_features = X.cols();
    
    std::cout << "perceptron test" <<std::endl;
    MatType X_train;
    MatType X_test;
    VecType y_train;
    VecType y_test;
    std::tie(X_train, X_test, y_train, y_test) = data::train_test_split<MatType, VecType>(X, y, 0.95);

    linear_model::Perceptron<double> perceptron;   
    perceptron.fit(X_train, y_train);

    VecType y_pred;
    y_pred = perceptron.predict(X_test);
    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;
    
}

