#include "../src/methods/rule_model/apriori.hpp"   
using namespace openml;

int main() {
    std::vector<std::vector<std::size_t>> X;
    data::loadtxt<std::size_t>("../dataset/input-1.txt", X);
    // X = {{2, 5, 4, 9}, {1, 6}, {7, 3}, {3, 2, 1, 8, 5}, {2, 4, 7, 9}};

    rule_model::Apriori<std::size_t> apriori;
    apriori.fit_transform(X);
    apriori.print_rules();
}


