#include "../src/methods/naive_bayes/multinomial_naive_bayes.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X{{2., 0., 0., 0., 1., 2., 3., 1.},
              {0., 0., 1., 0., 2., 1., 0., 0.},
              {0., 1., 0., 1., 0., 2., 1., 0.},
              {1., 0., 0., 2., 0., 1., 0., 1.},
              {2., 0., 0., 0., 1., 0., 1., 3.},
              {0., 0., 1., 2., 0., 0., 2., 1.},
              {0., 1., 1., 0., 0., 0., 1., 0.},
              {1., 2., 0., 1., 0., 0., 1., 1.},
              {0., 1., 1., 0., 0., 2., 0., 0.},
              {0., 0., 0., 0., 0., 0., 0., 0.},
              {0., 0., 1., 0., 1., 0., 1., 0.}};
    VecType y{{0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.}};

    
    // std::cout << X << std::endl;
    // std::cout << y << std::endl;

    naive_bayes::MultinomialNB<double> clf;

    clf.fit(X, y);

    MatType log_prob = clf.predict_log_prob(X);
    std::cout << "log_prob" << std::endl;
    std::cout << log_prob << std::endl;

    MatType prob = clf.predict_prob(X);
    std::cout << "prob" << std::endl;
    std::cout << prob << std::endl;

    VecType pred_y = clf.predict(X);
    std::cout << "pred_y" << std::endl;
    std::cout << pred_y << std::endl;

}
