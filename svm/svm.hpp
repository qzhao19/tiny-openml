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

#ifndef min
template <class T> static inline T min(T x, T y) { return (x < y) ? x : y; }
#endif

#ifndef max
template <class T> static inline T max(T x, T y) { return (x > y) ? x : y; }
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


class Cache {
public:

    /**
     * according the n_samples, we alllocate the memory space where have the same number of n_samples
     * Lru_head 因为尚没有head_t 中申请到内存，故双向链表指向自己。
     * 至于size的处理，先将原来的byte 数目转化为float 的数目，然后扣除L个node的内存数目
    */
    Cache(int n_samples_, long int size_);       
    ~Cache();


    /**
     * 该函数保证Node[index]中至少有len 个float 的内存，并且将可以使用的内存块的指针放在
     * data 指针中。返回值为申请到的内存。
    */
    int get_data(const int index, Qfloat **data, int len);


    /**
     * swap values from two head_t 
     * 交换Node[i] 和Node[j]的内容，先从双向链表中断开，交换后重新进入双向链表中。
    */
    void swap_index(int i, int j);                      


private:
    int n_samples;                  // the number of data samples
    long int size;                  // 指定的全部内存

    /**
     * 结构node用来记录所申请内存的指针，并记录长度，而且通过双向的指针，形成链表，增加寻址的速度
    */
    struct Node {
        Node *prev, *next;        
        Qfloat *data;
        int len;                    // 数据长度
    };


    /**
     * 变量指针，该指针记录程序中所申请的内存。
    */
    Node *head;
    Node lru_head;

    void lru_delete(Node *cur_node);
    void lru_insert(Node *cur_node);
};





#endif
