#include "../methods/logistic_regression/logistic_regression.hpp"

using namespace regression;
using namespace data;

int main() {

    arma::mat X;
    arma::vec y;
    std::tie(X, y) = data::loadtxt<double>("../../../dataset/horse_colic_train.txt");
    
    arma::mat X_train;
    arma::mat X_test;
    arma::vec y_train;
    arma::vec y_test;
    std::tie(X_train, X_test, y_train, y_test) = data::train_test_split(X, y);

    std::cout << X_train.n_rows << std::endl;
    std::cout << y_train.n_rows << std::endl;

    LogisticRegression lr;
    lr.fit(X_train, y_train);

    arma::vec y_pred;
    y_pred = lr.predict(X_test);

    std::cout <<"******************" << std::endl;
    std::cout << "predict: " << y_pred << std::endl;
    std::cout << "true: " << y_test << std::endl;

    double score = lr.score(y_test, y_pred);
    std::cout <<"******************" << std::endl;
    std::cout << score << std::endl;


    LogisticRegression lr2("lbfgs");
    lr2.fit(X_train, y_train);

    arma::vec y_pred2;
    y_pred2 = lr2.predict(X_test);

    std::cout <<"******************" << std::endl;
    std::cout << "predict: " << y_pred2 << std::endl;
    std::cout << "true: " << y_test << std::endl;

    double score2 = lr2.score(y_test, y_pred2);
    std::cout <<"******************" << std::endl;
    std::cout << score2 << std::endl;

    return 0;

}
