#include "../src/methods/neighbors/k_nearest_neighbors.hpp"   
using namespace openml;


int main() {
    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVecType = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using ColVecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using IdxMatType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;

    MatType X, X_train, X_test;
    ColVecType y, y_train, y_test, y_pred;

    data::loadtxt<MatType, ColVecType>("../dataset/iris.txt", X, y);

    X_train = X.topRows(125);
    X_test = X.bottomRows(25);
    y_train = y.topRows(125);
    y_test = y.bottomRows(25);

    tree::KDTree<double> kd_tree(X_train);
    neighbors::KNearestNeighbors<double> knn(
        10, 4, "kdtree", "euclidean"
    );

    knn.fit(X_train, y_train);

    knn.predict(X_test);

    
}
