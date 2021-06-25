#ifndef SVM_HPP_
#define SVM_HPP_
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cfloat>
#include <cstring>
#include <cstdarg>
#include "svm_node.hpp"
#include "svm_params.hpp"

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
    while (exponent > 0) {
        if ((exponent & 1) == 1) {
            retval = retval * base;
        }
        base *= base;
        exponent >>= 1;
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
     * Lru_head 因为尚没有node 中申请到内存，故双向链表指向自己。
     * 至于size的处理，先将原来的byte 数目转化为float 的数目，然后扣除L个node的内存数目
    */
    Cache(int n_samples_, long int size_);       
    ~Cache();

    /**
     * 该函数保证Node[index]中至少有len个float 的内存，并且将可以使用的内存块的指针放在
     * data 指针中。返回值为申请到的内存。
    */
    int get_data(const int index, Qfloat **data, int len);

    /**
     * swap values from two head_t 
     * 交换Node[i] 和Node[j]的内容，先从双向链表中断开，交换后重新进入双向链表中。
    */
    void swap_index(int i, int j);                      

private:
    /**
     * @params n_samples: the number of data samples 
     * @params size: 指定的全部内存
     * @params Node: 结构node用来记录所申请内存的指针，并记录长度，而且通过双向的指针，形成链表，增加寻址的速度
     * @params *head: 变量指针，该指针记录程序中所申请的内存, ，单块申请到的内存用struct Node来记录所申请内存的指针，并记录长度
     * @params lru_head: 双向链表的表头
     * 
    */
    int n_samples;                  
    long int size;                 
    struct Node {
        Node *prev, *next;        
        Qfloat *data;
        int len;            // 数据长度
    };
    Node *head;
    Node lru_head;

    /**break the connection from current location*/
    void lru_delete(Node *cur_node);

    /**Insert a new link, insert to the last position*/
    void lru_insert(Node *cur_node);
};


class QMatrix {
public:
    /**Pure virtual function*/
    virtual Qfloat *get_Q(int col, int len) const = 0;
    virtual double *get_QD() const = 0;
    virtual void swap_index(int i, int j) const = 0;
    virtual ~QMatrix() {};
}



class Kernel : public QMatrix {
public:
    /**构造函数。初始化类中的部分常量、指定核函数、克隆样本数据*/
    Kernel(int len, svm_node * const *X_, const svm_params &params);
    virtual ~Kernel();

    /**核函数, 只有在predict时才用到*/
    static double k_function(const svm_node *x, const svm_node *y, const svm_params &params);

    /**Pure virtual function*/
    virtual Qfloat *get_Q(int col, int len) const  = 0;
    virtual double *get_QD() const = 0;

    /**虚函数，x[i]和x[j]中所存储指针的内容。如果x_square 不为空，则交换相应的内容*/
    virtual void swap_index(int i, int j) const; 

protected:
    /**函数指针，根据相应的核函数类型，来决定所使用的函数。在计算矩阵Q 时使*/
    double (Kernel::*kernel_function)(int i, int j) const;

private:
    /**
     * @params **X: 用来指向样本数据，每次数据传入时通过克隆函数来实现，完全重新分配内存，主要是为处理多类着想。
     * @params *X-square: 使用RBF核使用
     * @params kernel_type: the kernel function type, including linear, ploynomial, rbf, sigmoid
     * @params degree: the same parameter with svm_params.degree
     * @params gamma: 
     * @params coef0: 
    */
    const svm_node **X;
    double *X_square;

    const int kernel_type;
    const double degree;
    const double gamma;
    const double coef0;

    static double dot(const svm_node *px, const svm_node *py);

    double linear_kernel(int i, int j) const;

    double poly_kernel(int i, int j) const;

    double rbf_kernel(int i, int j) const;

    double sigmoid_kernel(int i, int j) const;

    double kernel_precomputed(int i, int j) const;

};






#endif

