#include "../src/methods/neighbors/kd_tree.hpp"   
using namespace openml;


int main() {
    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVecType = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using ColVecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using IdxMatType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;

    MatType X, X_train, X_test;
    ColVecType y;

    data::loadtxt<MatType, ColVecType>("../dataset/iris.txt", X, y);

    X_train = X.topRows(125);
    X_test = X.bottomRows(25);
    
}
