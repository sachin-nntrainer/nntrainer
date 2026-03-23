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

#include <common_properties.h>
#include <optimizer_devel.h>

namespace nntrainer {

class MeZOEpsilon : public Property<float> {
public:
  static constexpr const char *key =
    "MeZO_epsilon";                /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

class MeZOLearningRate : public Property<float> {
public:
  static constexpr const char *key =
    "MeZO_learning_rate";          /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

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
  MeZO();

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override { return 0.0002; }

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

  /**
   * @brief Set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Get the epsilon value for perturbation
   * @return Perturbation magnitude
   */
  float getEpsilon() const { return std::get<MeZOEpsilon>(mezo_props).get(); }

  /**
   * @brief Get the learning rate
   * @return Learning rate value
   */
  float getLearningRate() const {
    return std::get<MeZOLearningRate>(mezo_props).get();
  }

  /**
   * @brief Update multiple weights using MeZO gradient estimation
   * @param weights Vector of weights to update
   * @param seed Random seed for reproducibility
   * @param loss_plus Loss with positive perturbation
   * @param loss_minus Loss with negative perturbation
   */
  void updateWeightsMeZO(std::vector<nntrainer::Tensor *> &weights, int seed,
                         float loss_plus, float loss_minus);

  static constexpr const char *type = "MeZO";

private:
  std::tuple<MeZOEpsilon, MeZOLearningRate>
    mezo_props; /**< MeZO epsilon and Learning rate */
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __MeZO_H__ */
