#ifndef SVM_UTILS_HPP_
#define SVM_UTILS_HPP_

#include "svm_node.hpp"
#include "svm_model.hpp"
#include "svm_params.hpp"
#include "svm_params.hpp"



// training dataset
struct svm_model *train(const struct svm_data *data, const struct svm_params *params);

// cross validation 
void cross_validation(const struct svm_data *data, const struct svm_params *params, int n_folds, double *y_true);

// svae svm training model 
int save_model(const struct svm_model *model, const char *model_filename);











#endif
