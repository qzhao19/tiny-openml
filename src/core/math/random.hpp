#ifndef CORE_MATH_RANDOM_HPP
#define CORE_MATH_RANDOM_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

class Random {

// private:
    // std::random_device rand;
    std::mt19937 generator_;

    std::uniform_real_distribution<double_t> real_dist;
    std::uniform_int_distribution<std::size_t> int_dist;


public:
    // Random(): Random(std::random_device()) {};








};

}
}

#endif /*CORE_MATH_RANDOM_HPP*/
