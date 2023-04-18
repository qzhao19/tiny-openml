#include "../src/methods/rule_model/apriori.hpp"   
using namespace openml;


int main() {

    using SpMatType = Eigen::SparseMatrix<std::size_t>;

    std::vector<std::vector<std::size_t>> X;
    data::loadtxt<std::size_t>("../dataset/input-1.txt", X);

    rule_model::Apriori<std::size_t> apriori;

    apriori.fit(X);

}


