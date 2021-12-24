#ifndef CORE_DATA_LOAD_DATA_HPP
#define CORE_DATA_LOAD_DATA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace data {

/**
 * load dataset from a txt file
 * @param fp String the given filepath 
 * 
 * @return tuple(X, y), X ndarray of shape (n_samples, n_features)
 *         y ndarray of shape (n_sample, )
*/
template<typename DataType>
std::tuple<arma::mat, arma::vec> loadtxt(const std::string &fp) {
    std::ifstream fin(fp);
    
    // first goal: Figure out how many columns there are
    // by reading in and parsing the first line of the file
    std::vector<std::vector<DataType>> data;
    std::string first_line;
    std::getline(fin, first_line);
    std::istringstream iss(first_line); // used to separate each element in the line
    
    DataType elem;
    while (iss >> elem){
        std::vector<DataType> row;
        data.push_back(row); // add empty sequence
        data.back().push_back(elem); // insert first element
    }
    
    // First line and all sequences are now created.
    // Now we just loop for the rest of the way.
    bool end = false;
    while (!end){
        for (size_t i = 0; i < data.size(); i++){
            DataType elem;
            if (fin >> elem){
                data[i].push_back(elem);
            }
            else{
                // end of data.
                // could do extra error checking after this
                // to make sure the columns are all equal in size
                end = true;
                break;
            } 
        }
    }

    arma::mat matrix(&data[0][0], data.size(), data[0].size(), false, false);
    std::size_t n_features = matrix.n_cols;

    arma::mat X = matrix.head_cols(n_features - 1);
    arma::vec y = matrix.tail_cols(1);
    
    return std::make_tuple(X, y);

}
}

#endif
