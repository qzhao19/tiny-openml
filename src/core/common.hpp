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
    const auto compare = [](value_t const &p1, value_t const &p2) {
        return p1.second < p2.second;
    };
    return *std::max_element(x.begin(), x.end(), compare);
};

/**
 * Stack arrays in sequence horizontally (column wise).
 * 
 * @param x1_x2 ndarray of shape (nrows, ncols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType hstack(const MatType& x1, const MatType& x2) {
    if (x1.rows() != x2.rows()) {
        throw std::invalid_argument("hstack with mismatching number of rows");
    }
    std::size_t nrows = x1.rows(), ncols1 = x1.cols(), ncols2 = x2.cols();
    MatType retval(nrows, (ncols1 + ncols2));
    retval << x1, x2;
    return retval;
};

/**
 * Stack arrays in sequence vertically (row wise). This is equivalent to concatenation 
 * along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N).
 * 
 * @param x1_x2 ndarray of shape (nrows, ncols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType vstack(const MatType& x1, const MatType& x2) {
    if (x1.cols() != x2.cols()) {
        throw std::invalid_argument("vstack with mismatching number of columns");
    }
    std::size_t ncols = x1.cols(), nrows1 = x1.rows(), nrows2 = x2.rows();
    MatType retval((nrows1 + nrows2), ncols);
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
    std::size_t nrows = x.rows(), ncols = x.cols();
    MatType trans_x = x.transpose();
    VecType flatten_vec(Eigen::Map<VecType>(trans_x.data(), nrows * ncols));
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
template<typename MatType, typename VecType, typename IdxVecType>
IdxVecType argmax(const MatType& x, int axis = 0) {
    if (axis == 1) {
        std::size_t nrows = x.rows();
        IdxVecType max_index{nrows};
        for (std::size_t i = 0; i < nrows; i++) {
            x.row(i).maxCoeff(&max_index[i]);
        }
        return max_index;
    }
    else if (axis == 0) {
        std::size_t ncols = x.cols();
        IdxVecType max_index{ncols};
        for (std::size_t j = 0; j < ncols; j++) {
            x.col(j).maxCoeff(&max_index[j]);
        }
        return max_index;
    }
    else if (axis == -1) {
        // flayyen a 2d matrix into 1d verctor
        VecType flatten_vec;
        flatten_vec = flatten<MatType, VecType>(x);
        
        // get the max index of flattened vector
        IdxVecType max_index{1};
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
template<typename MatType, typename VecType, typename IdxVecType>
IdxVecType argmin(const MatType& x, int axis = 0) {

    if (axis == 1) {
        std::size_t nrows = x.rows();
        IdxVecType min_index{nrows};
        for (std::size_t i = 0; i < nrows; i++) {
            x.row(i).minCoeff(&min_index[i]);
        }
        return min_index;
    }
    else if (axis == 0) {
        std::size_t ncols = x.cols();
        IdxVecType min_index{ncols};
        for (std::size_t j = 0; j < ncols; j++) {
            x.col(j).minCoeff(&min_index[j]);
        }
        return min_index;
    }
    else if (axis == -1) {
        VecType flatten_vec;
        flatten_vec = flatten<MatType, VecType>(x);
        
        // get the max index of flattened vector
        IdxVecType min_index{1};
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
template<typename MatType, typename IdxMatType, typename IdxVecType>
IdxMatType argsort(const MatType& x, int axis = 0, bool reverse = false) {
    // std::size_t nrows = x.rows(), ncols = x.cols();
    MatType copy_x = x;
    std::size_t nrows = x.rows(), ncols = x.cols();
    
    IdxMatType idxvec;
    IdxMatType idxmat(nrows, ncols);

    if (axis == 1) {
        idxvec.resize(1, ncols);
        idxvec = IdxVecType::LinSpaced(ncols, 0, ncols).transpose();
    }
    else if (axis == 0) {
        idxvec.resize(nrows, 1);
        idxvec = IdxVecType::LinSpaced(nrows, 0, nrows);
    }
    else {
        std::ostringstream err_msg;
        err_msg << "The axis " << axis << " is out of bounds "
                << "for array of dimension 2." << std::endl;
        throw std::invalid_argument(err_msg.str());
    }

    std::size_t idx = 0;
    
    if (!reverse) {
        if (axis == 1) {
            for (auto row : copy_x.rowwise()) {
                std::sort(idxvec.data(), idxvec.data() + idxvec.size(), 
                    [&row](std::size_t i, std::size_t j) -> bool {
                        return row(i) < row(j);
                    }
                );
                idxmat.row(idx) = idxvec;
                idx++;
            }
        }
        else if (axis == 0) {
            for (auto col : copy_x.colwise()) {
                std::sort(idxvec.data(), idxvec.data() + idxvec.size(), 
                    [&col](std::size_t i, std::size_t j) -> bool {
                        return col(i) < col(j);
                    }
                );
                idxmat.col(idx) = idxvec;
                idx++;
            }
        }
    }
    else {
        if (axis == 1) {
            for (auto row : copy_x.rowwise()) {
                std::sort(idxvec.data(), idxvec.data() + idxvec.size(), 
                    [&row](std::size_t i, std::size_t j) -> bool {
                        return row(i) > row(j);
                    }
                );
                idxmat.row(idx) = idxvec;
                idx++;
            }
        }
        else if (axis == 0) {
            for (auto col : copy_x.colwise()) {
                std::sort(idxvec.data(), idxvec.data() + idxvec.size(), 
                    [&col](std::size_t i, std::size_t j) -> bool {
                        return col(i) > col(j);
                    }
                );
                idxmat.col(idx) = idxvec;
                idx++;
            }
        }
    }
    
    return idxmat;
};

/**
 * Return a sorted copy of an array.
 * @param x ndarray
 *      Array to be sorted
 * @param axis int, default is 0
 *      Axis along which to sort. if axis = 0, along the row aixs
 *      if axis = 1, along the column
 * @param reverse bool, default is false
 *      if reverse is true, the sort is descending order, otherwise 
 *      sorts the list in the ascending order.
*/
template<typename MatType, typename DataType = typename MatType::value_type>
MatType sort(const MatType& x, int axis = 0, bool reverse = false) {
    // copy const matrix
    MatType copy_x = x;
    const auto asc = [](const DataType a, const DataType b) -> bool { 
        return a < b; 
    };

    const auto desc = [](const DataType a, const DataType b) -> bool { 
        return a > b; 
    };

    if (axis == 1) {
        if (!reverse) {
            for (auto row : copy_x.rowwise()) {
                std::sort(row.begin(), row.end(), asc);
            }
        }
        else {
            for (auto row : copy_x.rowwise()) {
                std::sort(row.begin(), row.end(), desc);
            }
        }
    }
    else if (axis == 0) {
        if (!reverse) {
            for (auto col : copy_x.colwise()) {
                std::sort(col.begin(), col.end(), asc);
            }
        }
        else {
            for (auto col : copy_x.colwise()) {
                std::sort(col.begin(), col.end(), desc);
            }
        }
    }
    else {
        std::ostringstream err_msg;
        err_msg << "The axis " << axis << " is out of bounds "
                << "for array of dimension 2." << std::endl;
        throw std::invalid_argument(err_msg.str());
    }

    return copy_x;
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
    std::size_t nrows = vec.size();
    std::size_t ncols = vec.at(0).size();

    using RowType = Eigen::Vector<DataType, Eigen::Dynamic>;

    MatType retmat(nrows, ncols);
    retmat.row(0) = RowType::Map(&vec[0][0], ncols);

    for (std::size_t i = 1; i < nrows; i++) {
        if (ncols != vec.at(i).size()) {
            std::ostringstream err_msg;
            err_msg << "vector[" << i << "] size = " << vec.at(i).size() 
                    << "does not match vector[0] size " << ncols << std::endl; 
            throw std::invalid_argument(err_msg.str());

        }
        retmat.row(i) = RowType::Map(&vec[i][0], ncols);
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
 * @param x vector of shape (nrows, 1)
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
template<typename VecType, typename IdxVecType>
IdxVecType where(const VecType& x) {
    std::vector<Eigen::Index> index_vec;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x(i)) {
            index_vec.push_back(i);
        }
    }
    Eigen::Map<IdxVecType> index(index_vec.data(), index_vec.size());
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
 * convert a string to integer
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
        // throw std::invalid_argument("k is more that number of array.");
        return std::vector<std::vector<DataType>>();
    }

    std::vector<DataType> copy_x = x;
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
 * @param v1 the source vector we want to compare
 * @param v2 the target vector to check if it is a subset
*/
template <typename DataType>
bool contains(const std::vector<DataType>& v1, 
    const std::vector<DataType>& v2){
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
 * @param v1 the target vector 1 
 * @param v2 the target vector 2
 * @return a vector of the two vectors
*/
template<typename DataType>
std::vector<DataType> merge(const std::vector<DataType>& v1,
    const std::vector<DataType>& v2){
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


/**
 * sorts the rows of a 2d vector in ascending or desc order based on the elements in the first column. 
 * When the first column contains repeated elements, sortrows sorts according to the values in the 
 * next column and repeats this behavior for succeeding equal values.
 * 
 * @param X, 2d vector to sort
 * @param reverse bool, default is false
 *      if reverse is true, the sort is descending order, otherwise 
 *      sorts the list in the ascending order.
*/
template <typename DataType>
std::vector<std::vector<DataType>> sort(
    const std::vector<std::vector<DataType>>& x, 
    int axis = 0, 
    bool reverse = false) {

    if (x.empty() || x[0].empty()) {
        throw std::invalid_argument("Input 2d array is empty!");
    }

    std::vector<std::vector<DataType>> copy_x = x;
    std::vector<std::vector<DataType>> result = x;    
    if (axis == 0){
        if (!reverse){
            for (size_t i = 0; i < copy_x[0].size(); i++) {
                // Sort column i
                std::sort(copy_x.begin(), copy_x.end(), [&](std::vector<DataType>& v1, std::vector<DataType>& v2) {
                    return (v1[i] < v2[i]); 
                });
                // Save the results of the sort of column i in the auxiliary vector  
                for (size_t j = 0; j < x.size(); ++j)
                    result[j][i] = copy_x[j][i];
            }
        }
        else {
            for (size_t i = 0; i < copy_x[0].size(); i++) {
                std::sort(copy_x.begin(), copy_x.end(), [&](std::vector<DataType>& v1, std::vector<DataType>& v2) {
                    return (v1[i] > v2[i]); 
                });
                for (size_t j = 0; j < copy_x.size(); ++j)
                    result[j][i] = copy_x[j][i];
            }
        }
    }
    else if (axis == 1) {
        const auto asc = [](const DataType a, const DataType b) -> bool { 
            return a < b; 
        };

        const auto desc = [](const DataType a, const DataType b) -> bool { 
            return a > b; 
        };
        if (!reverse){
            for (auto& r : copy_x) {
                std::sort(r.begin(), r.end(), asc);
            }
            result = copy_x;
        }
        else {
            for (auto& r : copy_x) {
                std::sort(r.begin(), r.end(), desc);
            }
            result = copy_x;
        }
    }
    else {
        std::ostringstream err_msg;
        err_msg << "The axis " << axis << " is out of bounds "
                << "for array of dimension 2." << std::endl;
        throw std::invalid_argument(err_msg.str());
    }
    return result;
};

/**
 * Go through a vector<vector<T> > and find the unique rows
 * have a value ind for each row that is 1/0 indicating if 
 * a value has been previously searched.
 * 
*/
template<typename DataType>
std::vector<std::vector<DataType>> remove_duplicate_rows(const std::vector<std::vector<DataType>>& X) {
    if (X.empty() || X[0].empty()) {
        throw std::invalid_argument("Input 2d vector is empty!");
    }

    std::size_t nrows = X.size();
    std::size_t num_searches = 1, cur_idx = 0, it = 1;
    std::vector<std::size_t> indices(nrows, 0);
    // create a deque to store the unique inds
    std::deque<std::size_t> unique_inds;  

    indices[cur_idx] = 1;
    unique_inds.emplace_back(0);
    while(num_searches < nrows) {
        if (it >= nrows) {
            // find next non-duplicate value, push back
            ++cur_idx;
            while (indices[cur_idx]) {
                ++cur_idx;
            }
            unique_inds.emplace_back(cur_idx);
            ++num_searches;
            
            // start search for duplicates at the next row
            it = cur_idx + 1; 
            if (it >= nrows && nrows == num_searches) {
                break;
            }
        }
        
        if (!indices[it] && X[cur_idx]==X[it]) {
            indices[it] = 1;
            ++num_searches;
        }
        ++it;
    } 

    std::vector<std::vector<DataType>> result;
    // loop over the deque and push the unique vectors    
    std::deque<std::size_t>::iterator iter;
    const std::deque<size_t>::iterator end = unique_inds.end();
    result.reserve(unique_inds.size());
    for(iter = unique_inds.begin(); iter != end; ++iter) {
        result.push_back(X[*iter]);
    }
    return result;
};

/**
 * Determine array equality
 * @param x 1d vector
 * @param y 1d vector
 * @return logical true if A and B are equivalent; otherwise, it returns logical false
*/
template<typename DataType>
bool is_equal(const std::vector<DataType>& x, const std::vector<DataType>& y) {
    auto pair = std::mismatch(x.begin(), x.end(), y.begin());
    return (pair.first == x.end() && pair.second == y.end());
};

/**
 * sorts the rows of a matrix based on the elements in the first column. 
 * When first column contains repeated elements, sortrows sorts according 
 * to the values in the next column and repeats this behavior for succeeding equal values.
 * 
 * @param x input data 2d vector
 * @param reverse bool, default is false
 *      if reverse is true, the sort is descending order, otherwise 
 *      sorts the list in the ascending order.
*/
template<typename DataType>
std::vector<std::vector<DataType>> sortrows(
    const std::vector<std::vector<DataType>>& x, 
    bool reverse = false) {
    
    if (x.empty() || x[0].empty()) {
        throw std::invalid_argument("Input 2d array is empty!");
    }
    std::vector<std::vector<DataType>> copy_x = x;

    const auto asc = [](const std::vector<DataType>& v1, const std::vector<DataType>& v2) {
        std::size_t index = 0;
        while (index < v1.size()) {
            if (v1[index] == v2[index]) {
                ++index; 
            } 
            else {
                return v1[index] < v2[index];
            }
        }
    };
    const auto desc = [](const std::vector<DataType>& v1, const std::vector<DataType>& v2) {
        std::size_t index = 0;
        while (index < v1.size()) {
            if (v1[index] == v2[index]) {
                ++index; 
            } 
            else {
                return v1[index] > v2[index];
            }
        }
    };

    if (!reverse) {
        std::sort(copy_x.begin(), copy_x.end(), asc);
    }
    else {
        std::sort(copy_x.begin(), copy_x.end(), desc);
    }
    return copy_x;
};

/**
 * Get the difference between two vectors, given 2 sorted vectors v1 and v2, 
 * returns a vector with the elements of v1 that are not on v2.
*/
template<typename DataType>
std::vector<DataType> difference(const std::vector<DataType>& v1, 
    const std::vector<DataType>& v2, 
    bool sorted = false) {

    std::vector<DataType> copy_v1 = v1, copy_v2 = v2;
    static const auto is_nan = [] (DataType v) { 
        return std::isnan(v); 
    } ;
    assert(std::none_of(copy_v1.begin(), copy_v1.end(), is_nan) &&
           std::none_of(copy_v2.begin(), copy_v2.end(), is_nan));

    if (sorted) {
        std::sort(copy_v1.begin(), copy_v1.end());
        std::sort(copy_v2.begin(), copy_v2.end());
    }
    
	std::vector<DataType> result ;
	std::set_difference(copy_v1.begin(), std::unique(copy_v1.begin(), copy_v1.end()),
                        copy_v2.begin(), copy_v2.end(), std::back_inserter(result));
	return result;
};


}
}
#endif /*CORE_COMMON_HPP*/
