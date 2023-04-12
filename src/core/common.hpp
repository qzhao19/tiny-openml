#ifndef CORE_COMMON_HPP
#define CORE_COMMON_HPP
#include "../prereqs.hpp"

namespace openml {
namespace common {

/**
 *find the maximum value in a std::map and return the corresponding std::pair
*/
template <typename Container>
auto max_element(Container const &x)
    -> typename Container::value_type {
    
    using value_t = typename Container::value_type;
    const auto compare = [](value_t const &p1, value_t const &p2)
    {
        return p1.second < p2.second;
    };
    return *std::max_element(x.begin(), x.end(), compare);
};

/**
 * Stack arrays in sequence horizontally (column wise).
 * 
 * @param x1_x2 ndarray of shape (num_rows, num_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType hstack(const MatType& x1, const MatType& x2) {
    if (x1.rows() != x2.rows()) {
        throw std::invalid_argument("hstack with mismatching number of rows");
    }
    std::size_t num_rows = x1.rows(), num_cols1 = x1.cols(), num_cols2 = x2.cols();
    MatType retval(num_rows, (num_cols1 + num_cols2));
    retval << x1, x2;
    return retval;
};

/**
 * Stack arrays in sequence vertically (row wise). This is equivalent to concatenation 
 * along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N).
 * 
 * @param x1_x2 ndarray of shape (num_rows, num_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType vstack(const MatType& x1, const MatType& x2) {
    if (x1.cols() != x2.cols()) {
        throw std::invalid_argument("vstack with mismatching number of columns");
    }
    std::size_t num_cols = x1.cols(), num_rows1 = x1.rows(), num_rows2 = x2.rows();
    MatType retval((num_rows1 + num_rows2), num_cols);
    retval << x1, 
              x2;
    return retval;
};

/**
 * flatten a 2D vector of vector to a one dimension vector
*/
template<typename DataType, typename = typename DataType::value_type>
DataType flatten(const std::vector<DataType>& v) {
    return std::accumulate(v.begin(), v.end(), DataType{}, [](auto& dest, auto& src) {
        dest.insert(dest.end(), src.begin(), src.end());
        return dest;
    });
};

/**
 * flatten a matrix of 2d to a vector
 * 
 * @param x Eigen matrix type ndarray 
 * @return one dim vector of Eigrn type
*/
template<typename MatType, typename VecType>
VecType flatten(const MatType& x) {
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    MatType trans_x = x.transpose();
    VecType flatten_vec(Eigen::Map<VecType>(trans_x.data(), num_rows * num_cols));
    return flatten_vec;
};

/**
 * Repeat elements of an matrix.
 * 
 * @param x Input array.
 * @param repeats int. The number of repetitions for each element. 
 * @param axis int. The axis along which to repeat values. 
 * 
 * @return  Output array which has the same shape as a, except along the given axis.
*/
template<typename MatType>
MatType repeat(const MatType& x, 
    int repeats, int axis) {
    
    MatType retval;
    if (axis == 0) {
        retval = x.colwise().replicate(repeats);
    }
    else if (axis == 1) {
        retval = x.rowwise().replicate(repeats);
    }
    return retval;
};

/**
 * Returns the indices of the maximum values along an axis.
 *
 * @param x 2darray of input data
 * @param axis int, the given specified axis.
 * 
 * @return the indices of the maximum values along an axis
 *      if axis = 0, return index vector of max element along horizontal
 *      if axis = 1, retunn index vector of max element along vertical
 *      if axis = -1, return the max index scalar element of 2d 
*/
template<typename MatType, typename VecType, typename IdxType>
IdxType argmax(const MatType& x, int axis = 0) {
    if (axis == 1) {
        std::size_t num_rows = x.rows();
        IdxType max_index{num_rows};
        for (std::size_t i = 0; i < num_rows; i++) {
            x.row(i).maxCoeff(&max_index[i]);
        }
        return max_index;
    }
    else if (axis == 0) {
        std::size_t num_cols = x.cols();
        IdxType max_index{num_cols};
        for (std::size_t j = 0; j < num_cols; j++) {
            x.col(j).maxCoeff(&max_index[j]);
        }
        return max_index;
    }
    else if (axis == -1) {
        // flayyen a 2d matrix into 1d verctor
        VecType flatten_vec;
        flatten_vec = flatten<MatType, VecType>(x);
        
        // get the max index of flattened vector
        IdxType max_index{1};
        flatten_vec.maxCoeff(&max_index[0]);

        return max_index;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};

/**
 * Returns the indices of the minimum values along an axis.
 * @param x 2darray of input data
 * @param axis int, the given specified axis.
 * 
 * @return the indices of the minimum values along an axis
 *      if axis = 0, return index vector of min element along horizontal
 *      if axis = 1, retunn index vector of min element along vertical
 *      if axis = -1, return the min index scalar element of 2d 
*/
template<typename MatType, typename VecType, typename IdxType>
IdxType argmin(const MatType& x, int axis = 0) {

    if (axis == 1) {
        std::size_t num_rows = x.rows();
        IdxType min_index{num_rows};
        for (std::size_t i = 0; i < num_rows; i++) {
            x.row(i).minCoeff(&min_index[i]);
        }
        return min_index;
    }
    else if (axis == 0) {
        std::size_t num_cols = x.cols();
        IdxType min_index{num_cols};
        for (std::size_t j = 0; j < num_cols; j++) {
            x.col(j).minCoeff(&min_index[j]);
        }
        return min_index;
    }
    else if (axis == -1) {
        VecType flatten_vec;
        flatten_vec = flatten<MatType, VecType>(x);
        
        // get the max index of flattened vector
        IdxType min_index{1};
        flatten_vec.minCoeff(&min_index[0]);

        return min_index;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};

/**
 * Returns the indices that would sort an array.
 * Perform an indirect sort along the given axis
 * It returns an array of indices of the same shape 
 * as a that index data along the given axis in sorted order.
 * 
 * @param x ndarray like data
 *      array to sort
 * @param axis int, default 1
 *      Axis along which to sort. 
 * @param order string, default 'asc'
 *      his argument specifies which fields to compare first
 * @return index array
 *      Array of indices that sort a along the specified axis
*/
template<typename AnyType, typename IdxType>
IdxType argsort(const AnyType& x, int axis = 1, std::string order = "asc") {
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    IdxType index ;
    if (axis == 1) {
        index = IdxType::LinSpaced(num_rows, 0, num_rows);
    }
    else if (axis == 0) {
        index = IdxType::LinSpaced(num_cols, 0, num_cols);
    }

    if (order == "asc") {
        std::stable_sort(index.data(), index.data() + index.size(), 
            [&x](std::size_t i, std::size_t j) -> bool {
                return x(i, 0) < x(j, 0);
            }
        );
    }
    else if (order == "desc") {
        std::stable_sort(index.data(), index.data() + index.size(), 
            [&x](std::size_t i, std::size_t j) -> bool {
                return x(i, 0) > x(j, 0);
            }
        );
    }
    else {
        std::ostringstream err_msg;
        err_msg << "Invalid given sort order " << order.c_str() << std::endl;
        throw std::out_of_range(err_msg.str());
    }

    return index;
};

/**
 * Element-wise minimum of array elements.
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType fmin(const MatType& x, const DataType y) {
    return (x.array() >= y).select(y, x);
}

/**
 * convert a 2d-array of std vector vector into an eigen matrix
 * vector<vector<T>> --> Matrix 
 * 
 * @param vec 2d-array of vector vector 
 *      input 2d vector
 * @return an eigen matrix type 
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType vec2mat(std::vector<std::vector<DataType>> vec) {
    std::size_t num_rows = vec.size();
    std::size_t num_cols = vec.at(0).size();

    using RowType = Eigen::Vector<DataType, Eigen::Dynamic>;

    MatType retmat(num_rows, num_cols);
    retmat.row(0) = RowType::Map(&vec[0][0], num_cols);

    for (std::size_t i = 1; i < num_rows; i++) {
        if (num_cols != vec.at(i).size()) {
            std::ostringstream err_msg;
            err_msg << "vector[" << i << "] size = " << vec.at(i).size() 
                    << "does not match vector[0] size " << num_cols << std::endl; 
            throw std::invalid_argument(err_msg.str());

        }
        retmat.row(i) = RowType::Map(&vec[i][0], num_cols);
    }

    return retmat;
};

/**
 * override vec2mat function
*/
template<typename VecType, 
    typename DataType = typename VecType::value_type>
VecType vec2mat(std::vector<DataType> vec) {
    std::size_t num_elems = vec.size();
    Eigen::Map<VecType> retvec(vec.data(), num_elems);
    return retvec;
}

/**
 * Find the unique elements of an 1d array.
 * @param x vector of shape (num_rows, 1)
 *      input vector
 * @return a tuple of (VecType, VecType)
 *    returns the unsorted unique elements of an array and 
 *    the indices of the input array that give the unique values
*/
template<typename VecType, 
    typename DataType = typename VecType::value_type>
std::tuple<VecType, VecType> unique(const VecType& x){

    if (x.cols() != 1) {
        throw std::out_of_range("Rows of vector should be equal to 1.");
    }

    std::vector<DataType> stdvec_index;
    std::vector<DataType> stdvec_value;

    // convert the input vector to std vector
    std::vector<DataType>  stdvec_x;
    for (std::size_t i = 0; i < x.rows(); i++) {
        stdvec_x.push_back(x(i, 0));
    }

    std::sort(stdvec_x.begin(), stdvec_x.end());
    auto first = std::begin(stdvec_x), last = std::end(stdvec_x);
    std::set<std::size_t> hash_set;
    std::map<DataType, std::size_t> hash_map;
    for(std::size_t i = 0; first != last; ++i, ++first){
        auto iter_pair = hash_map.insert(std::make_pair(*first, i));
        if(iter_pair.second){
            stdvec_value.push_back(iter_pair.first->first);
            hash_set.insert(iter_pair.first->second);
            hash_set.insert(i);
        }
    }
    stdvec_index = {hash_set.begin(), hash_set.end()};

    VecType retvec_value = Eigen::Map<VecType>(stdvec_value.data(), stdvec_value.size(), 1);
    VecType retvec_index = Eigen::Map<VecType>(stdvec_index.data(), stdvec_index.size(), 1);

    return std::make_tuple(retvec_value, retvec_index);
};

/**
 * Return index of elements chosen depending on condition.
 * @param x condition array_like, bool
 * @return An array with index of elements 
*/
template<typename VecType, typename IdxType>
IdxType where(const VecType& x) {
    std::vector<Eigen::Index> index_vec;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x(i)) {
            index_vec.push_back(i);
        }
    }
    Eigen::Map<IdxType> index(index_vec.data(), index_vec.size());
    return index;
};

/**
 * limite the values in an array.
 * @param x array_like, array containing elements to clip.
 * @param max_min array value type,  maximum and minimum value
*/
template<typename MatType, typename DataType = typename MatType::value_type>
MatType clip(const MatType& x, DataType max, DataType min) {
    return x.array().min(max).max(min);
};

/**
 * convert a string to size_t
 * @param s string, input string to convert
*/
template <class DataType>  
DataType string_to_integer(const std::string& s){  
    std::istringstream iss(s);  
    DataType num;  
    iss >> num;  
    return num;      
};

/**
 * Get all non-repeating combinations of numbers (permutations)
 * 
 * @param x input list
 * @param k the length subsequences
 * @return k length subsequences of elements from the input
*/
template <typename DataType>
std::vector<std::vector<DataType>> combinations(const std::vector<DataType>& x, std::size_t k) {
    if (x.size() < k) {
        throw std::invalid_argument("k is more that number of array.");
    }

    std::vector<DataType> copy_x = x;
    if (!std::is_sorted(copy_x.begin(), copy_x.end())) {
        std::sort(copy_x.begin(), copy_x.end());
    }

    std::vector<bool> bitset(k, 1);
    bitset.resize(copy_x.size(), 0);

    std::vector<std::vector<DataType>> combs;
    do {
        std::vector<DataType> tmp;
        for (std::size_t i = 0; i != x.size(); ++i) {
            if (bitset[i]) {
                tmp.emplace_back(copy_x[i]);
            }
        }
        combs.emplace_back(tmp);
    } while (std::prev_permutation(bitset.begin(), bitset.end()));

    return combs;
};


/**
 * check whether a vector is a subvector of another
 * @param v2 the target vector to check if it is a subset
 * @param v1 the source vector we want to compare
*/
template <typename DataType>
bool contains(const std::vector<DataType>& v1, const std::vector<DataType>& v2){
    for (typename std::vector<DataType>::const_iterator i = v1.begin(); i != v1.end(); i++){
        bool found = false;
        for (typename std::vector<DataType>::const_iterator j = v2.begin(); j != v2.end(); j++){
            if (*i == *j){
                found = true;
                break;
            }
        }
        if (!found){
            return false;
        }
    }
    return true;
};

/**
 * merge two vector of different size
*/
template<typename DataType>
std::vector<DataType> merge(const std::vector<DataType>& v1,
    const std::vector<DataType>& v2)
{
    std::vector<DataType> result;
    result.reserve(v1.size() + v2.size());
    auto ait = v1.begin(), bit = v2.begin();
    
    // copy while both have more elements:
    for(; ait != v1.end() && bit != v2.end(); ++ait, ++bit) {
        result.push_back(*ait);
        result.push_back(*bit);
    }

    // copy the rest
    if(ait != v1.end()) {
        result.insert(result.end(), ait, v1.end());
    } 
    else if(bit != v2.end()) {
        result.insert(result.end(), bit, v2.end());
    }

    return result;
};


}
}
#endif /*CORE_COMMON_HPP*/
