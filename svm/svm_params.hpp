#ifndef SVM_PARAMS_HPP_
#define SVM_PARAMS_HPP_


// svm type + kernel function type
enum {C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR};
enum {LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED};

struct svm_params {
    int svm_type;
    int kernel_type;
    double degree;     //degree of the polynomial kernel function
    double gamma;      //for ploy & rbf & sigmoid, Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
    double coff0;      //for ploy & sigmoid, Independent term in kernel function

    /*for training only*/
    double cache_size;
    double epsilons;    //stopping criteria, for SVR
    double C;           //Regularization parameter
    double n_weights;   //the number of weight, in this case, default value is 0, for svm_binary_svc_probabiliy, is 2
    int *weight_label;
    double *weight;
    
    double nu;          //边际误差分数的上限和支持向量分数相对于训练样本总数的下限
    double p;           //设置 epsilon - SVR 中损失函数的值
    int shrinking;       //是否使用启发式，0或1
    int probability;    //do probability estimation


};

#endif /*SVM_PARAMS_HPP_*/
