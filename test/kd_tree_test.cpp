#include "../src/core/tree/kd_tree.hpp"   
using namespace openml;


int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    data::loadtxt<double>("../dataset/iris.txt", X, y);

    tree::KDTree<double> kd_tree(X);

    kd_tree.test();

}






