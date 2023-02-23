#include "../src/methods/decomposition/truncated_svd.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    decomposition::TruncatedSVD<double> truncated_svd;

    truncated_svd.fit(X);

    std::cout << "transform data" << std::endl;
    std::cout << truncated_svd.transform(X) << std::endl;

    std::cout << "data explained_variance" << std::endl;
    std::cout << truncated_svd.get_explained_var() << std::endl;

    std::cout << "singular_values_" << std::endl;
    std::cout << truncated_svd.get_singular_values() << std::endl;

    std::cout << "components" << std::endl;
    std::cout << truncated_svd.get_components() << std::endl;

}
