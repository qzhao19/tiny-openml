#include "linear_regression.hpp"
#include <iostream>

using namespace regression;


int main() {

    arma::mat X_train(50, 6, arma::fill::randu);

    arma::vec y_train(50, arma::fill::randu);


    LinearRegression lr;

    lr.fit(X_train, y_train);

    arma::vec theta = lr.get_theta();

    std::cout <<"******************" << std::endl;
    std::cout << theta << std::endl;
    std::cout << theta.n_rows << std::endl;


    arma::mat X_test(10, 6, arma::fill::randu);

    arma::vec y_pred(10);

    lr.predict(X_test, y_pred);

    std::cout <<"******************" << std::endl;
    std::cout << y_pred << std::endl;

    return 0;

}
