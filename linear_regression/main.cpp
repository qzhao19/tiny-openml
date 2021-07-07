#include "linear_regression.hpp"
#include <iostream>

using namespace regression;


int main() {

    arma::mat X(50, 6, arma::fill::randu);

    arma::vec y(50, arma::fill::randu);


    std::cout << X << std::endl;

    std::cout <<"******************" << std::endl;

    std::cout << y << std::endl;


    LinearRegression lr;

    lr.fit(X, y);

    // arma::vec theta = lr.get_theta();

    // std::cout <<"******************" << std::endl;

    // std::cout << theta << std::endl;


    return 0;

}