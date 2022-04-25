#ifndef METHODS_TREE_BASE_HPP
#define METHODS_TREE_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace decision_tree {

template<typename DataType>
class DecisionTree {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
protected:
     /**
     * choose a threshold from a given features vector, first we sort
     * this sample vector, then find the unique threshold
    */
    const VecType choose_feature_threshold(const VecType& x) const {
        std::size_t num_samples = x.rows();
        IdxType sorted_index = utils::argsort<VecType, IdxType>(x);
        VecType sorted_x = x(sorted_index);
        
        std::vector<DataType> stdvec_x;
        for (std::size_t i = 1; i < num_samples; ++i) {
            DataType mean = (sorted_x(i - 1) + sorted_x(i)) / 2;
            stdvec_x.push_back(mean);
        }

        std::sort(stdvec_x.begin(), stdvec_x.end());
        auto last = std::unique(stdvec_x.begin(), stdvec_x.end());
        stdvec_x.erase(last, stdvec_x.end());
        // VecType unique_x_values;
        Eigen::Map<VecType> unique_x(stdvec_x.data(), stdvec_x.size());

        return unique_x;
    }

    /**
     * Iterate through all unique values of feature column i and 
     * calculate the impurity. If this threshold resulted in a 
     * higher information gain than previously recorded save 
     * the threshold value and the feature index
    */
    const std::tuple<double, std::size_t, DataType> best_split(const MatType& X, 
            const VecType& y) const {
        
        std::size_t num_features = X.cols();
        
        std::size_t best_feature_index = ConstType<std::size_t>::max();
        DataType best_feature_value = ConstType<DataType>::quiet_NaN();
        double best_impurity = ConstType<double>::min();

        for (std::size_t feature_index = 0; feature_index < num_features; ++feature_index) {
            VecType feature_values;
            feature_values = choose_feature_threshold(X.col(feature_index));
            for (auto& threshold : feature_values) {
                std::vector<std::size_t> left_index, right_index;
                std::tie(left_index, right_index) = data::divide_on_feature<MatType>(X, feature_index, threshold);

                VecType left_y, right_y;
                left_y = y(left_index);
                right_y = y(right_index);

                double impurity = this->compute_impurity(y, left_y, right_y);
                // std::cout << "impurity = " << impurity << std::endl;
                if (impurity > best_impurity) {
                    best_impurity = impurity;
                    best_feature_index = feature_index;
                    best_feature_value = threshold;
                }
            }
        }
        return std::make_tuple(best_impurity, best_feature_index, best_feature_value);
    }

    /** pure virtual function to compute the inpurity */
    virtual const double compute_impurity (
        const VecType& y, 
        const VecType& left_y, 
        const VecType& right_y) const = 0;

    // virtual const std::tuple<VecType, DataType> compute_node_value(
    //     const VecType& y) const = 0;

public:
    DecisionTree() {};
    ~DecisionTree() {};

};

} // decision_tree
} // openml

#endif /*METHODS_TREE_BASE_HPP*/
