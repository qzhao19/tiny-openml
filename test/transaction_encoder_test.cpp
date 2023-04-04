#include "../src/core/preprocessing/transaction_encoder.hpp"   
using namespace openml;

int main() {

    using SpMatType = Eigen::SparseMatrix<std::size_t>;

    std::vector<std::vector<std::size_t>> X;
    data::loadtxt<std::size_t>("../dataset/input-1.txt", X);

    preprocessing::TransactionEncoder<std::size_t> transaction_encoder;

    SpMatType sp_mat = transaction_encoder.fit_transform(X);

    std::cout << sp_mat << std::endl;

}
