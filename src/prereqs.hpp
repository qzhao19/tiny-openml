#ifndef PREREQS_HPP
#define PREREQS_HPP

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <queue>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stack>
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

// add Eigen sparse module 
#include <eigen3/Eigen/SparseCore>


#ifndef EIGEN_MAX_CPP_VER
  #define EIGEN_MAX_CPP_VER 14
#endif


#ifndef min
template <typename T> static inline T min(T x,T y) { return (x < y) ? x : y;}
#endif
#ifndef max
template <typename T> static inline T max(T x,T y) { return (x > y) ? x : y;}
#endif

// #include <eigen3/Eigen/Dense>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>

#endif /*PREREQS_HPP*/
