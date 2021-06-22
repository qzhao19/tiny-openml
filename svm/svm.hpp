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
    // according the n_samples, we alllocate the memory space where have the same n_samples
    Cache(int n_samples_, int size_);       
    ~Cache();

    int get_data(const int index, Qfloat **data, int len);


    // swap values from two head_t 
    void swap_index();                      


private:
    int n_samples;                  // the number of data samples
    int size;                       // 指定的全部内存

    //结构head_t用来记录所申请内存的指针，并记录长度，而且通过双向的指针，形成链表，增加寻址的速度
    struct head_t {
        head_t *prev, *next;        
        Qfloat *data;
        int len;
    };

    // 变量指针，该指针记录程序中所申请的内存。
    head_t *head;

    // the head node of doubled linked list
    head_t lru_head;

    // 从双向链表中删去某个元素的链接，一般是删去当前所指向的元素
    void lru_delete(head_t *h);

    // 在链表后面插入一个新的链接
    void lru_insert(head_t *h);




};







#endif

