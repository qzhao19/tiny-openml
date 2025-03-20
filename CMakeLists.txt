cmake_minimum_required(VERSION 3.14)

project(tiny-openml VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(ML_HEADERS
    src/core/common.hpp
    src/core/data/load.hpp
    src/core/data/split.hpp
    src/core/loss/hinge_loss.hpp
    src/core/loss/huber_loss.hpp
    src/core/loss/log_loss.hpp
    src/core/loss/mean_squared_error.hpp
    src/core/loss/softmax_loss.hpp
    src/core/math/extmath.hpp
    src/core/math/linalg.hpp
    src/core/math/ops.hpp
    src/core/math/proba.hpp
    src/core/math/random.hpp
    src/core/math/shuffle.hpp
    src/core/metric.hpp
    src/core/optimizer/base.hpp
    src/core/optimizer/lbfgs/lbfgs.hpp
    src/core/optimizer/lbfgs/params.hpp
    src/core/optimizer/lbfgs/search_policies/backtracking.hpp
    src/core/optimizer/lbfgs/search_policies/base.hpp
    src/core/optimizer/lbfgs/search_policies/bracketing.hpp
    src/core/optimizer/sgd/decay_policies/exponential_decay.hpp
    src/core/optimizer/sgd/decay_policies/step_decay.hpp
    src/core/optimizer/sgd/sag.hpp
    src/core/optimizer/sgd/scd.hpp
    src/core/optimizer/sgd/sgd.hpp
    src/core/optimizer/sgd/tg.hpp
    src/core/optimizer/sgd/update_policies/momentum_update.hpp
    src/core/optimizer/sgd/update_policies/nesterov_momentum_update.hpp
    src/core/optimizer/sgd/update_policies/vanilla_update.hpp
    src/core/preprocessing/transaction_encoder.hpp
    src/core/preprocessing.hpp
    src/core/tree/hash_tree.hpp
    src/core/tree/kd_tree.hpp
    src/core.hpp
    src/methods/cluster/kmeans.hpp
    src/methods/cluster/spectral_clustering.hpp
    src/methods/decomposition/base.hpp
    src/methods/decomposition/pca.hpp
    src/methods/decomposition/truncated_svd.hpp
    src/methods/linear_model/base.hpp
    src/methods/linear_model/lasso_regression.hpp
    src/methods/linear_model/linear_regression.hpp
    src/methods/linear_model/logistic_regression.hpp
    src/methods/linear_model/perceptron.hpp
    src/methods/linear_model/ridge_regression.hpp
    src/methods/linear_model/softmax_regression.hpp
    src/methods/mixture/gaussian_mixture.hpp
    src/methods/naive_bayes/base.hpp
    src/methods/naive_bayes/gaussian_naive_bayes.hpp
    src/methods/naive_bayes/multinomial_naive_bayes.hpp
    src/methods/neighbors/base.hpp
    src/methods/neighbors/k_nearest_neighbors.hpp
    src/methods/rule_model/apriori.hpp
    src/methods/tree/base.hpp
    src/methods/tree/decision_tree_classifier.hpp
    src/methods/tree/decision_tree_regressor.hpp
    src/prereqs.hpp
)

add_library(tiny-openml INTERFACE)

tiny-openmltarget_sources(tiny-openml INTERFACE ${ML_HEADERS})


enable_testing()

file(GLOB TEST_SOURCES "test/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} tiny-openml gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()