#ifndef SVM_MODEL_H_
#define SVM_MODEL_H_
#include "svm_node.h"
#include "svm_params.h"



struct svm_model {

    /**/
    svm_params parms;       //the number of parameters in training step
    int n_classes;          //the number of classes
    int n_support_vec;      //the number of support vectors
    svm_node **support_vec; //support vector

    double **W;             //the weights, coefficients for SVs in decision functions
    double *b;              //the biases, constants in decision functions

    double *prob_A;
    double *prob_B;


    
    int *c_lable;           //label for each class label[i]
    int *n_c_support_vec;   //number of suport vector for each class n_s_v[i], nSV[0] + nSV[1] + ... + nSV[k-1] = l

    int free_support_vec;


};

#endif /*SVM_MODEL_H_*/