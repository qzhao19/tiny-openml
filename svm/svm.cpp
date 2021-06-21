#include "svm.hpp"


#ifndef min
template <class T> static inline T min(T x, T y) {
    return (x < y) ? x : y;
}
#endif


#ifndef min
template <class T> static inline T max(T x, T y) {
    return (x > y) ? x : y;
}
#endif


template<class T> static inline void swap(T &x, T &y) {
    T tmp;
    tmp = x;
    x = y;
    y = tmp;
}

template<class S, class T> static inline void clone(T*& dst, S*& src, int n){
    dst = new T[n];
    memcpy((void *)dst, (void *)src, sizeof(T)*n);
}


static inline double powi(double base, int exponent) {
    double retval = 1.0;
    long long exponent1 = exponent;

    while (exponent1 > 0) {
        if ((exponent1 & 1) == 1) {
            retval = retval * base;
        }
        base *= base;
        exponent1 >>= 1;
    }

    return retval;
}


#define INF MAX_VAL
#define TAU 1e-12
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))


static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

