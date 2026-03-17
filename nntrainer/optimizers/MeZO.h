// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   MeZO.h
 * @date   17 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the MeZO optimizer.
 */
#ifndef __MeZO_H__
#define __MeZO_H__
#ifdef __cplusplus

#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @class   MeZO optimizer class
 * @brief   MeZO (Zero'th order) optimizer class
 */
class MeZO : public Optimizer {
public:
  /**
   * @brief Construct a new MeZO object
   *
   */
  MeZO() {}

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override { return 0.0001; }

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override { return; }

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return MeZO::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim) override {
    return {};
  }

  static constexpr const char *type = "MeZO";
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __MeZO_H__ */