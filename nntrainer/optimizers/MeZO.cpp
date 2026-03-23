// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   MeZO.cpp
 * @date   17 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the MeZO optimizer.
 */

#include <MeZO.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

MeZO::MeZO() : mezo_props(MeZOEpsilon(), MeZOLearningRate()) {

  auto &[epsilon, learningRate] = mezo_props;
  epsilon.set(0.01f);
  learningRate.set(0.0001f);
}

void MeZO::setProperty(const std::vector<std::string> &values) {

  auto remain_props = loadProperties(values, mezo_props);

  if (!remain_props.empty()) {
    std::string msg = "[MeZO Optimizer] Unknown Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void MeZO::updateWeightsMeZO(std::vector<nntrainer::Tensor *> &weights,
                             int seed, float loss_plus, float loss_minus) {
  std::mt19937 gen;
  gen.seed(seed);
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  float mezo_epsilon = getEpsilon();
  float lr = getLearningRate();
  float projected_grad = (loss_plus - loss_minus) / (2.0f * mezo_epsilon);

  for (const auto *ptr : weights) {

    float *data = (float *)(ptr->getData());
    size_t param_size = ptr->getDim().getDataLen();

    for (size_t i = 0; i < param_size; ++i) {

      data[i] += lr * normal_dist(gen) * projected_grad;
    }
  }
}

} // namespace nntrainer
