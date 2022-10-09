#include "../src/methods/linear_model/perceptron.hpp"
using namespace openml;


int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);


    std::size_t num_features = X.cols();
    
    std::cout << "perceptron test" <<std::endl;
    
}

