#include "../src/core/loss/log_loss.hpp"
#include "../src/core/optimizer/lbfgs/lbfgs.hpp"
#include "../src/core/optimizer/lbfgs/params.hpp"

using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);

    std::size_t num_features = X.cols();
    std::cout << "lbfgs test" <<std::endl;
    
    VecType w(num_features);
    w.setZero();
    VecType opt_w(num_features);

    loss::LogLoss<double> log_loss;
    optimizer::LineSearchParams<double> ls_params;

    optimizer::LBFGS<double, 
        loss::LogLoss<double>, 
        optimizer::LineSearchParams<double>> lbfgs(w, log_loss, ls_params);
    
    lbfgs.optimize(X, y);
    opt_w = lbfgs.get_coef();
    
    std::cout << opt_w <<std::endl;

}

