#ifndef PREREQS_HPP
#define PREREQS_HPP

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

template<typename DataType>
using ConstType = std::numeric_limits<DataType>;


#ifdef EIGEN_USE_BLAS
  #include <eigen3/Eigen/src/misc/blas.h>
#endif

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/SVD>


#ifndef EIGEN_MAX_CPP_VER
  #define EIGEN_MAX_CPP_VER 14
#endif


// #include <eigen3/Eigen/Dense>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>

#endif /*PREREQS_HPP*/
