#include "../src/methods/tree/decision_tree_classifier.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    // std::cout << X << std::endl;
    decision_tree::DecisionTreeClassifier<double> clf;

    // clf.test_func(X, y);
    clf.fit(X, y);

    std::cout << "predict prob" << std::endl;
    MatType pred_prob;
    pred_prob = clf.predict_prob(X);
    std::cout << pred_prob << std::endl;

    std::cout << "predict label" << std::endl;
    VecType y_pred;
    y_pred = clf.predict(X);
    std::cout << y_pred << std::endl;


    return 0;
}