#include "../src/methods/decomposition/pca.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    decomposition::PCA<double> pca;

    pca.fit(X);

    std::cout << "data covariance" << std::endl;
    std::cout << pca.get_covariance() << std::endl;

    std::cout << "precision matrix" << std::endl;
    std::cout << pca.get_precision() << std::endl;

    std::cout << "score" << std::endl;
    std::cout << pca.score(X) << std::endl;

    std::cout << "transform data" << std::endl;
    std::cout << pca.transform(X) << std::endl;

}
