#include "linear_regression.hpp"

using namespace regression;


int main() {

    arma::mat X_train(50, 6, arma::fill::randu);

    arma::vec y_train(50, arma::fill::randu);

    LinearRegression lr;

    lr.fit(X_train, y_train);

    arma::mat X_test(10, 6, arma::fill::randu);
    arma::vec y_test(10, arma::fill::randu);


    arma::vec y_pred;
    y_pred = lr.predict(X_test);

    std::cout <<"******************" << std::endl;
    std::cout << y_pred << std::endl;

    double score = lr.score(y_test, y_pred);

    std::cout <<"******************" << std::endl;
    std::cout << score << std::endl;

    return 0;

}
