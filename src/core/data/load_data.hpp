#ifndef CORE_DATA_LOAD_DATA_HPP
#define CORE_DATA_LOAD_DATA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

template<typename DataType>
void loadtxt(const std::string &filename, arma::mat& retmat) {
    std::ifstream fin(filename);
    
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

    arma::mat retmat_(&data[0][0], data.size(), data[0].size(), false, false);
    retmat = retmat_;
}



#endif
