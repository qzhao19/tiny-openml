#ifndef METHOD_RULE_MODEL_APRIORI_HPP
#define METHOD_RULE_MODEL_APRIORI_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace rule_model {

template<typename DataType>
class Apriori {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    

};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/