#ifndef CORE_OPTIMIZER_SGD_SGD_HPP
#define CORE_OPTIMIZER_SGD_SGD_HPP

namespace openml {
namespace optimizer {

template<typename DataType>
class SGD {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double tol;
    double alpha;
    bool shuffle;
    std::size_t max_iters;
    std::size_t batch_size;
    
public:

    template<typename LossFuncionType, 
        typename UpdatePolicyType, 
        typename DecayPolicytype>
    void optimize(LossFuncionType& loss_fn, 
        UpdatePolicyType& update_policy ,
        DecayPolicytype& decay_policy ,
        VecType& weight) {

        std::size_t num_samples = loss_fn.get_num_samples();
        std::size_t num_features = loss_fn.get_num_features();

        VecType grad(num_features);

        for (std::size_t i = 0; i < max_iters; i++) {
            if (shuffle) {
                loss_fn.shuffle();
            }
            std::size_t num_iter = num_samples / batch_size;

            for (std::size_t j = 0; j < num_iter; j++) {
                std::size_t begin = j * batch_size;
                
                
                loss_fn.gradient(weight, grad, begin, batch_size);

                weight = weight - alpha * grad; 

                double loss_val = log_loss.evaluate(W, begin, batch_size);

                // double error = ;


                // if (abs_mat.array().abs() < tol) {
                //     break;
                // }



            }
        }

    }

};

}
}
#endif
