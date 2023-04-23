#ifndef CORE_OPTIMIZER_LBFGS_LBFGS_HPP
#define CORE_OPTIMIZER_LBFGS_LBFGS_HPP
#include "./search_policies/base.hpp"
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
#include "../base.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

/**
 * LBFGS algorithm
*/
template<typename DataType, 
    typename LossFuncionType,
    typename LineSearchParamType>
class LBFGS: public BaseOptimizer<DataType, 
    LossFuncionType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    double delta_; 
    std::string linesearch_policy_;
    std::size_t mem_size_;
    std::size_t past_;
    LineSearchParamType linesearch_params_;

public:
    LBFGS(const VecType& x0,
        const LossFuncionType& loss_func,
        const LineSearchParamType& linesearch_params,
        const std::size_t max_iter = 0, 
        const std::size_t mem_size = 8, 
        const std::size_t past = 3,
        const double tol = 1e-5,
        const double delta = 1e-6,
        const std::string linesearch_policy = "backtracking",
        const bool shuffle = true,
        const bool verbose = true): BaseOptimizer<DataType, 
            LossFuncionType>(x0, 
                loss_func, 
                max_iter, 
                tol,
                shuffle, 
                verbose),
            linesearch_params_(linesearch_params),
            linesearch_policy_(linesearch_policy),
            mem_size_(mem_size),
            past_(past),
            delta_(delta) {};
    
    ~LBFGS() {};

    void optimize(const MatType& X, 
        const VecType& y) {
        
        // define the initial parameters
        VecType x = this->x0_;
        std::size_t num_dims = x.rows();
        std::size_t i, j, k, end, bound;
        double fx, ys, yy, rate, beta;

        // intermediate variables: previous x, gradient, previous gradient, directions
        VecType xp(num_dims);
        VecType g(num_dims);
        VecType gp(num_dims);
        VecType d(num_dims);

        // an array for storing previous values of the objective function
        VecType pfx(std::max(static_cast<std::size_t>(1), past_));

        // define step search policy
        std::unique_ptr<BaseLineSearch<DataType, LossFuncionType, LineSearchParamType>> linesearch;
        if (linesearch_policy_ == "backtracking") {
            linesearch = std::make_unique<LineSearchBacktracking<DataType, LossFuncionType, LineSearchParamType>>(
                X, y, this->loss_func_, linesearch_params_
            );
        }
        else if (linesearch_policy_ == "bracketing") {
            linesearch = std::make_unique<LineSearchBracketing<DataType, LossFuncionType, LineSearchParamType>>(
                X, y, this->loss_func_, linesearch_params_
            );
        }
        else {
            throw std::invalid_argument("Cannot find line search policy.");
        }

        // Initialize the limited memory variables
        MatType mem_s(num_dims, mem_size_);
        MatType mem_y(num_dims, mem_size_);
        VecType mem_ys(mem_size_);
        VecType mem_alpha(mem_size_);

        // Evaluate the function value and its gradient
        fx = this->loss_func_.evaluate(X, y, x);
        g = this->loss_func_.gradient(X, y, x);

        // Store the initial value of the cost function
        pfx(0) = fx;

        // Compute the direction we assume the initial hessian matrix H_0 as the identity matrix
        d = -g;

        // make sure the intial points are not sationary points (minimizer)
        double xnorm = static_cast<double>(x.norm());
        double gnorm = static_cast<double>(g.norm());

        if (xnorm < 1.0) {
            xnorm = 1.0;
        }
        if (gnorm / xnorm <= this->tol_) {
            std::cout << "ERROR: The initial variables already minimize the objective function" << std::endl;
            return ;
        }

        // compute intial step = 1.0 / norm2(d)
        double step = 1.0 / static_cast<double>(d.norm());

        k = 1;
        end = 0;
        bound = 0;
        while (true) {
            // store current x and gradient value
            xp = x;
            gp = g;

            // apply line search function to find optimized step, search for an optimal step
            int ls = linesearch->search(x, fx, g, d, step, xp, gp);
            
            if (ls < 0) {
                x = xp;
                g = gp;
                std::cout << "ERROR: lbfgs exit: the point return to the privious point." << std::endl;
                break ;
            }

            // Convergence test -- gradient
            // criterion is given by the following formula:
            // ||g(x)|| / max(1, ||x||) < tol
            xnorm = static_cast<double>(x.norm());
            gnorm = static_cast<double>(g.norm());

            if (this->verbose_) {
                std::cout << "Iteration = " << k << ", fx = " << fx 
                          << ", xnorm value = " << xnorm 
                          << ", gnorm value = " << gnorm << std::endl;
            }

            if (xnorm < 1.0) {
                xnorm = 1.0;
            }
            if (gnorm / xnorm <= this->tol_) {
                std::cout << "INFO: success to reached convergence (tol)." << std::endl;
                break;
            }
            
            // Convergence test -- objective function value
            // The criterion is given by the following formula:
            // |f(past_x) - f(x)| / max(1, |f(x)|) < delta.
            if (past_ <= k) {
                rate = std::abs(pfx(k % past_) - fx) / std::max(std::abs(fx), 1.0);
                if (rate < delta_) {
                    std::cout << "INFO: success to meet stopping criteria (ftol)." << std::endl;
                    break;
                }
                pfx(k % past_) = fx;
            }
            if ((this->max_iter_ != 0) && (this->max_iter_ < k + 1)) {
                std::cout << "INFO: the algorithm routine reaches the maximum number of iterations" << std::endl;
                break;
            }

            // Update vectors s and y:
            // s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
            // y_{k+1} = g_{k+1} - g_{k}.
            mem_s.col(end) = x - xp;
            mem_y.col(end) = g - gp;

            // Compute scalars ys and yy:
            // ys = y^t \cdot s = 1 / \rho.
            // yy = y^t \cdot y.
            // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            ys = mem_y.col(end).dot(mem_s.col(end));
            yy = mem_y.col(end).dot(mem_y.col(end));
            mem_ys(end) = ys;

            // Compute the negative of gradients
            d = -g;

            bound = (mem_size_ <= k) ? mem_size_ : k;
            ++k;
            end = (end + 1) % mem_size_;
            j = end;

            // Loop 1
            for (i = 0; i < bound; ++i) {
                // if (--j == -1) j = m-1
                j = (j + mem_size_ - 1) % mem_size_;
                // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
                mem_alpha(j) = mem_s.col(j).dot(d) / mem_ys(j);
                // \q_{i} = \q_{i+1} - \alpha_{i} y_{i}
                d.noalias() += (-mem_alpha(j)) * mem_y.col(j);
            }

            d *= ys / yy;

            // loop 2
            for (i = 0; i < bound; ++i) {
                /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamm_{i}. */
                beta = mem_y.col(j).dot(d) / mem_ys(j);
                /* \gamm_{i+1} = \gamm_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
                d.noalias() += (mem_alpha(j) - beta) * mem_s.col(j);
                /* if (++j == m) j = 0; */
                j = (j + 1) % mem_size_; 
            }

            step = 1.0;  
        }
        this->opt_x_ = x;
    }

};

}
}
#endif /*CORE_OPTIMIZER_LBFGS_LBFGS_HPP*/
