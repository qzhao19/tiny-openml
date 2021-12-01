#ifndef CORE_UTILITY_UTILS_HPP
#define CORE_UTILITY_UTILS_HPP
#include "../../prereqs.hpp"

namespace utils {

/**
 *find the maximum value in a std::map and return the corresponding std::pair
*/
template <class Container>
auto max_element(Container const &x)
    -> typename Container::value_type {
    
    using value_t = typename Container::value_type;
    const auto compare = [](value_t const &p1, value_t const &p2)
    {
        return p1.second < p2.second;
    };
    return *std::max_element(x.begin(), x.end(), compare);
}




};
#endif
