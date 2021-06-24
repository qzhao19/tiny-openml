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

/**
 * break the connection from current losaction and 
*/
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


