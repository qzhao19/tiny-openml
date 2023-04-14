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

    using FreqItemsetsType = std::map<std::size_t, 
        std::pair<std::vector<std::vector<DataType>>, std::vector<DataType>>>;
    double min_support_;
    double min_confidence_;


protected:
    const std::vector<std::vector<DataType>> generate_candidates(
        const FreqItemsetsType& all_frequent, 
        std::size_t k) {
        
         
        std::vector<std::vector<DataType>> candidates;
        if (k == 1) {
            candidates = common::combinations<DataType>(all_frequent[0], k + 1);
        }
        elif (k == 2) {
            
        }





    }




public:

    Apriori(): min_support_(0.4), min_support_(0.6) {};
    Apriori(double min_support, double min_confidence): 
        min_support_(min_support), 
        min_confidence_(min_confidence) {};
    
    ~Apriori() {};

};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/