#ifndef SVM_PARAMS_HPP_
#define SVM_PARAMS_HPP_
/**
 * Kernenle function type
 * Linear kernel function : K(x_i, x_j) = transpose(x_i) * x_j
 * polynomiale kernel function : K(x_i, x_j) = pos((gamme * transpose(x_i) * X_j + r), d)
 * RBF kernel function: K(x_i, x_j) = exp(-gamma * norm(x_i - x_j, 2))
 * sigmoid kernenl function: tanh(gamma * (transpose(x_i) * x_j) + r)
*/


// svm type + kernel function type
enum {C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR};
enum {LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED};

struct svm_params {
    int svm_type;       // svm type
    int kernel_type;    // kernel function type
    double degree;      //degree of the polynomial kernel function
    double gamma;       //for ploy & rbf & sigmoid, Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
    double coff0;       //for ploy & sigmoid, Independent term in kernel function

    /*for training only*/
    double cache_size;  // 制定训练所需要的内存
    double epsilons;    //stopping criteria, for SVR
    double C;           //Regularization parameter
    double n_weights;   //the number of weight, in this case, default value is 0, for svm_binary_svc_probabiliy, is 2
    int *weight_label;  // 权重，元素个数由n_weights决定
    double *weight;
    
    double nu;          //边际误差分数的上限和支持向量分数相对于训练样本总数的下限
    double p;           //设置 epsilon - SVR 中损失函数的值
    int shrinking;       //是否使用启发式，0或1
    int probability;    //do probability estimation


};

#endif /*SVM_PARAMS_HPP_*/
