#ifndef METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP
#define METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {

template<typename DataType>
class DecisionTreeClassifier : public DecisionTree<DataType> {

private:
    
    std::string criterion;
    std::size_t min_samples_split;
    std::size_t max_depth; 
    double min_impurity;

protected:
    const double compute_impurity(const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) {
        
        std::size_t num_damples = y.rows();
        sad::size_t left_num_samples = left_y.rows();
        std::size_t right_num_samples = right_y.rows();

        if (criterion == "gini") {

            





        }









    }

public:
    DecisionTreeClassifier(): DecisionTree<DataType>(), 
        criterion("entropy") {};


    explicit DecisionTreeClassifier(std::string criterion_): 
        DecisionTree<DataType>(), 
        criterion(criterion_) {};

    DecisionTreeClassifier(std::string criterion_, 
        std::size_t min_samples_split_,
        std::size_t max_depth_,
        double min_impurity_): 
            decision_tree::DecisionTree<DataType>(
                min_samples_split_, 
                max_depth_, 
                min_impurity_),
            criterion(criterion_) {};


    void test_func(const MatType& X, const VecType& y){
        this->best_split(X, y);
    }




};

} // decision_tree
} // openml

#endif /*METHODS_TREE_DECISION_TREE_CLASSIFIER_HPP*/

