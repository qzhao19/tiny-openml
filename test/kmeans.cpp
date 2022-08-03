#include "../src/methods/cluster/kmeans.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);
    
    cluster::KMeans<double> kmeans;

    kmeans.fit(X);

    VecType y_pred = kmeans.predict(X);

    std::cout << "y_pred" << std::endl;
    std::cout << y_pred << std::endl;


    return 0;
}
