#include "../../../pca.hpp"

using namespace pca;

int main() {
    arma::mat X;
    arma::vec y;
    std::tie(X, y) = data::loadtxt<double>("../../../dataset/iris.txt");
    
    std::cout <<"*********start SVD PCA*********" << std::endl;
    PCA svd_pca;
    svd_pca.fit(X);
    arma::mat X_trains_2 = svd_pca.transform(X);
    std::cout << X_trains_2 << std::endl;
    std::cout << svd_pca.score(X) << std::endl;
    std::cout << svd_pca.get_covariance() << std::endl;

    return 0;

}
