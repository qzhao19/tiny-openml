#ifndef CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_LINE_SEARCH_BACKTRACKING_HPP
#define CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_LINE_SEARCH_BACKTRACKING_HPP

namespace openml {
namespace optimizer {

template <typename DataType>
class LineSearchBacktracking {

template <typename DataType>
class LineSearchBacktracking
{
private:
    typedef Eigen::Matrix<DataType, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<DataType>& param,
                           const Vector& xp, const Vector& drt, const DataType& step_max,
                           DataType& step, DataType& fx, Vector& grad, DataType& dg, Vector& x)
    {
        // Decreasing and increasing factors
        const DataType dec = 0.5;
        const DataType inc = 2.1;

        // Check the value of step
        if (step <= DataType(0))
            throw std::invalid_argument("'step' must be positive");

        // Save the function value at the current x
        const DataType fx_init = fx;
        // Projection of gradient on the search direction
        const DataType dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (dg_init > 0)
            throw std::logic_error("the moving direction increases the objective function value");

        const DataType test_decr = param.ftol * dg_init;
        DataType width;

        int iter;
        for (iter = 0; iter < param.max_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            fx = f(x, grad);

            if (fx > fx_init + step * test_decr || (fx != fx))
            {
                width = dec;
            }
            else
            {
                // Armijo condition is met
                if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
                    break;

                const DataType dg = grad.dot(drt);
                if (dg < param.wolfe * dg_init)
                {
                    width = inc;
                }
                else
                {
                    // Regular Wolfe condition is met
                    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
                        break;

                    if (dg > -param.wolfe * dg_init)
                    {
                        width = dec;
                    }
                    else
                    {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            if (step < param.min_step)
                throw std::runtime_error("the line search step became smaller than the minimum value allowed");

            if (step > param.max_step)
                throw std::runtime_error("the line search step became larger than the maximum value allowed");

            step *= width;
        }

        if (iter >= param.max_linesearch)
            throw std::runtime_error("the line search routine reached the maximum number of iterations");
    }
};


};

}
}

#endif /* CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_LINE_SEARCH_BACKTRACKING_HPP */