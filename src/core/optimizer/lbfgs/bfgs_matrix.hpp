#ifndef CORE_OPTIMIZER_LBFGS_BFGS_MATRIX_HPP
#define CORE_OPTIMIZER_LBFGS_BFGS_MATRIX_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType>
class BFGSMatrix {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
    using RefConstVecType = Eigen::Ref<const VecType>;
    // typedef std::vector<int> IndexSet;

    int m_m;         // Maximum number of correction vectors
    DataType m_theta;  // theta * I is the initial approximation to the Hessian matrix
    MatType m_s;      // History of the s vectors
    MatType m_y;      // History of the y vectors
    VecType m_ys;     // History of the s'y values
    VecType m_alpha;  // Temporary values used in computing H * v
    int m_ncorr;     // Number of correction vectors in the history, m_ncorr <= m
    int m_ptr;       // A Pointer to locate the most recent history, 1 <= m_ptr <= m
                     // Details: s and y vectors are stored in cyclic order.
                     //          For example, if the current s-vector is stored in m_s[, m-1],
                     //          then in the next iteration m_s[, 0] will be overwritten.
                     //          m_s[, m_ptr-1] points to the most recent history,
                     //          and m_s[, m_ptr % m] points to the most distant one.


public:
    // Constructor
    BFGSMatrix() {}
    ~BFGSMatrix() {}


    // Reset internal variables
    // n: dimension of the vector to be optimized
    // m: maximum number of corrections to approximate the Hessian matrix
    inline void reset(int n, int m) {
        m_m = m;
        m_theta = Scalar(1);
        m_s.resize(n, m);
        m_y.resize(n, m);
        m_ys.resize(m);
        m_alpha.resize(m);
        m_ncorr = 0;
        m_ptr = m;  // This makes sure that m_ptr % m == 0 in the first step
    }


    // Add correction vectors to the BFGS matrix
    inline void add_correction(const RefConstVecType& s, const RefConstVecType& y){
        const int loc = m_ptr % m_m;

        m_s.col(loc).noalias() = s;
        m_y.col(loc).noalias() = y;

        // ys = y's = 1/rho
        const DataType ys = m_s.col(loc).dot(m_y.col(loc));
        m_ys[loc] = ys;

        m_theta = m_y.col(loc).squaredNorm() / ys;

        if (m_ncorr < m_m)
            m_ncorr++;

        m_ptr = loc + 1;
    }

    inline void apply_Hv(const Vector& v, const Scalar& a, Vector& res){
        res.resize(v.size());

        // L-BFGS two-loop recursion

        // Loop 1
        res.noalias() = a * v;
        int j = m_ptr % m_m;
        for (int i = 0; i < m_ncorr; i++)
        {
            j = (j + m_m - 1) % m_m;
            m_alpha[j] = m_s.col(j).dot(res) / m_ys[j];
            res.noalias() -= m_alpha[j] * m_y.col(j);
        }

        // Apply initial H0
        res /= m_theta;

        // Loop 2
        for (int i = 0; i < m_ncorr; i++)
        {
            const Scalar beta = m_y.col(j).dot(res) / m_ys[j];
            res.noalias() += (m_alpha[j] - beta) * m_s.col(j);
            j = (j + 1) % m_m;
        }
    }
};

}
}

#endif  // CORE_OPTIMIZER_LBFGS_BFGS_MATRIX_HPP