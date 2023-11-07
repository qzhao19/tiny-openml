#include "../src/core/tree/kd_tree.hpp"   
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
    tree::KDTree<double> kd_tree(X_train);
    
    MatType distances1;
    IdxMatType indices1;
    std::tie(distances1, indices1) = kd_tree.query(X_test, 4);
    std::cout << "distances1" << std::endl;
    std::cout << distances1 << std::endl;
    std::cout << "indices1" << std::endl;
    std::cout << indices1 << std::endl;

    // std::vector<std::pair<double, std::size_t>> knn;
    std::vector<std::vector<std::pair<double, std::size_t>>> knn;
    knn = kd_tree.query_radius(X_test, 0.5);
    for (auto nn : knn) {
        for (auto n : nn) {
            std::cout << n.first << " " << n.second << " ";
        }
        std::cout << std::endl;
    }
}
