#ifndef CORE_HPP
#define CORE_HPP

// #include "../src/methods/base.hpp"

#include "../src/core/utils.hpp"

#include "../src/core/math/ops.hpp"
#include "../src/core/math/shuffle.hpp"
#include "../src/core/math/linalg.hpp"
#include "../src/core/math/extmath.hpp"
#include "../src/core/math/dist.hpp"
#include "../src/core/math/random.hpp"


#include "../src/core/data/load.hpp"
#include "../src/core/data/split.hpp"

#include "../src/core/loss/hinge_loss.hpp"
#include "../src/core/loss/log_loss.hpp"
#include "../src/core/loss/mean_squared_error.hpp"

#include "../src/core/optimizer/base.hpp"
#include "../src/core/optimizer/sgd/sgd.hpp"
#include "../src/core/optimizer/sgd/sag.hpp"
#include "../src/core/optimizer/sgd/truncated_gradient.hpp"
#include "../src/core/optimizer/sgd/scd.hpp"
#include "../src/core/optimizer/lbfgs/lbfgs.hpp"


#include "../src/core/metric.hpp"




#endif /*CORE_HPP*/
