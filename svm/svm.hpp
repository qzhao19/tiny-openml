#ifndef SVM_HPP_
#define SVM_HPP_
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cfloat>
#include <cstring>
#include <cstdarg>
#include "svm_utils.hpp"

typedef float Qfloat;
typedef signed char schar; 


class Cache {
public:
    Cache(int n_samples_, int size_);
    ~Cache();


private:
    int n_samples;
    int size;
    struct head_t {
        head_t *prev, *next;
        Qfloat *data;
        int len;
    };

    



};







#endif

