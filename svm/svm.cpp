#include <iostream>
#include "svm.hpp"


Cache::Cache(int n_samples_, long int size_): n_samples(n_samples_), size(size_) {
    // assigne the number of element and the size of each element 
    // head = bytes number
    head = (Node *)calloc(n_samples, sizeof(Node));

    // convert the bytes number into the float number 
    size /= sizeof(Qfloat);

    // substract the total Node number (n_samples)
    size -= n_samples * sizeof(Node) / sizeof(Qfloat);
    size = max(size, 2 * (long int)n_samples);

    // lru_head didn't yet allocate the memory space, so it points itself 
    lru_head.next = lru_head.prev = &lru_head;

}

Cache::~Cache() {
    for (Node *cur_node = lru_head.next; cur_node != &lru_head; cur_node = cur_node -> next) {
        free(cur_node -> data);
    }
    free(head);
}

void Cache::lru_delete(Node *cur_node) {
    /**
     * Delete the link of an element from the doubly linked list without 
     * deleting or releasing the memory involved in the element. 
     * 
     *  node1 <-----> node2 <------> node3 <-----> node4
     * cur_node = node2
     * node1 -> next = node3
     * node3 -> prev = node1
     * 
    */
    cur_node -> prev -> next = cur_node -> next;
    cur_node -> next -> prev = cur_node -> prev;
}

void Cache::lru_insert(Node *cur_node) {
    /**
     * Insert a new link, insert to the last position
    */
    
    cur_node -> next = &lru_head;
    cur_node -> prev = lru_head.prev;
    cur_node -> prev -> next = cur_node;
    cur_node -> next -> prev = cur_node;
}

int Cache::get_data(const int index, Qfloat **data, int len) {
    /**
     * 该函数保证head_t[index]中至少有len 个float 的内存，
     * 并且将可以使用的内存块的指针放在data 指针中。返回值为申请到的内存.
     * 函数首先将head_t[index]从链表中断开,如果head_t[index]原来没有分配内存，则跳过断开这步。
     * 计算当前head_t[index]已经申请到的内存，如果不够，释放部分内存
    */
    
    Node *cur_node = &head[index];
    if (cur_node -> len) {
        lru_delete(cur_node);
    }

    int cur_len = len - cur_node -> len;
    if (cur_len > 0) {
        // free old space

        // size为所指定的全部内存
        while (size < cur_len) {
            Node *old_node = lru_head.next;
            lru_delete(old_node);
            free(old_node -> data);
            old_node -> data = 0;
            old_node -> len = 0;
        }

        // allocate new space, 把内存扩大到len
        cur_node -> data = (Qfloat *)realloc(cur_node -> data, sizeof(Qfloat) * len);
        size -= cur_len;
        swap(cur_node -> len , len);
    }

    lru_insert(cur_node);
    *data = cur_node -> data;

    return len;
}

void Cache::swap_index(int i, int j) {
    /**
     * 交换head_t[i] 和head_t[j]的内容，先从双向链表中断开，交换后重新进入双向链表中
    */
    if (i == j) {
        return ;
    }

    // break up link 
    if (head[i].len) lru_delete(&head[i]);
    if (head[j].len) lru_delete(&head[j]);

    // swap data
    swap(head[i].data, head[j].data);
    swap(head[i].len, head[j].len);

    // re-insert into list
    if (head[i].len) lru_insert(&head[i]);
    if (head[j].len) lru_insert(&head[j]);

    if (i > j) {    
        return ;
    }

    // ???
    for (Node *cur_node = lru_head.next; cur_node != &lru_head; cur_node = cur_node -> next) {
        if (cur_node -> len > i) {
            if (cur_node -> len > j) {
                swap(cur_node -> data[i], cur_node -> data[j]);
            }
            else {
                lru_delete(cur_node);
                free(cur_node -> data);
                size += cur_node -> len;
                cur_node -> data = 0;
                cur_node -> len = 0;
            }
        }
    }
}



Kernel::Kernel(int len, 
               svm_node *const *X_, 
               const svm_params &params) : kernel_type(params.kernel_type), 
                                           degree(params.degree), 
                                           gamma(params.gamma), 
                                           coef0(params.coef0) {                            
    switch (kernel_type)
    {
    case LINEAR:
        kernel_function = &Kernel::linear_kernel;
        break;
    
    case POLY:
        kernel_function = &Kernel::poly_kernel;
        break;

    case RBF:
        kernel_function = &Kernel::rbf_kernel;
        break;
    
    case SIGMOID:
        kernel_function = &Kernel::sigmoid_kernel;
        break;
    
    case PRECOMPUTED:
        kernel_function = &Kernel::kernel_precomputed;
        break;
    }

    clone(X, X_, len);

    if (kernel_type == RBF) {
        X_square = new double[len];
        for (int i = 0; i < len; i++) {
            X_square[i] = dot(X[i], X[i]);
        }
    }
    else {
        X_square = 0;
    }

}

Kernel::~Kernel() {
    delete[] X;
    delete[] X_square;
}


void Kernel::swap_index(int i, int j) const {
    /**虚函数,x[i]和 x[j]中所存储指针的内容。如果 x_square 不为空,则交换相应的内容*/
    swap(X[i], X[j]);
    if (X_square) {
        swap(X_square[i], X_square[j]);
    }
}

double Kernel::k_function(const svm_node *x, const svm_node *y, 
                          const svm_params &params) {

    /**其中 RBF 部分很有讲究。因为存储时,0 值不保留。如果所有 0 值都保留,第一个 while就可以都做完了;
     * 如果第一个 while 做不完,在 x,y 中任意一个出现 index = -1,第一个 while就停止,
     * 剩下的代码中两个 while 只会有一个工作,该循环直接把剩下的计算做完
     */
    switch (params.kernel_type) {
        case LINEAR:
            return dot(x, y);

        case POLY:
            return powi(params.gamma * dot(x, y) + params.coef0, params.degree);

        case RBF:
            double sum = 0.0;
            while (x -> index != -1 && y -> index != -1) {
                if (x -> index == y -> index) {
                    double minus_val = x -> value - y -> value;
                    sum += minus_val * minus_val;
                    ++x;
                    ++y;
                }

                if (x -> index > y -> index) {
                    sum += y -> value * y -> value;
                    ++y;
                }
                else {
                    sum += x -> value * x -> value;
                    ++x;
                }
            }

            while (x -> index != -1) {
                sum += x -> value * x -> value;
                ++x;
            }

            while (y -> index != -1) {
                sum += y -> value * y -> value;
                ++y;
            }

            return exp(-params.gamma * sum);


        case SIGMOID:
            return tanh(params.gamma * dot(x, y) + params.coef0);

        case PRECOMPUTED:
            return x[(int)(y -> value)].value;

        default:
            return 0;
    }
    
}


double Kernel::dot(const svm_node *px, const svm_node *py) {
    /**
     * 点乘两个样本数据，按svm_node 中index (一般为特征)进行运算，一般来说，index 中1，2，… 直到-1。返回点乘总和。
     * 例如：x1 = {1,2,3} , x2 = {4, 5, 6} 总和为sum = 1*4 + 2*5 + 3*6 ;在svm_node[3]中存储index = -1 时，停止计算。
    */
    double retval = 0.0;
    while ((px -> index != -1) && (py -> index != -1)) {
        
        if (px -> index == py -> index) {
            retval = px -> value * py -> value;
            ++px;
            ++py;
        } 
        else {
            if (px -> index > py -> index) {
                ++py;
            } 
            else {
                ++px;
            }
        }
    }
}

double Kernel::linear_kernel(int i, int j) const {
        // K(x_i, x_j) = transpose(x_i) * x_j 
        return dot(X[i], X[j]);
    };

double Kernel::poly_kernel(int i, int j) const {
    // K(x_i, x_j) = pow((gamme * transpose(x_i) * X_j + r), d)
    return powi(gamma * dot(X[i], X[j]) + coef0, degree);
};

double Kernel::rbf_kernel(int i, int j) const {
    // K(x_i, x_j) = exp(-gamma * norm(x_i - x_j, 2))
    return exp(-gamma * (X_square[i] + X_square[j] - 2 * dot(X[i], X[j])));
};

double Kernel::sigmoid_kernel(int i, int j) const {
    // tanh(gamma * (transpose(x_i) * x_j) + r)
    return tanh(gamma * dot(X[i], X[j]) + coef0);
};

double Kernel::kernel_precomputed(int i, int j) const {
    return X[i][(int)(X[j][0].value)].value;
}








