// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   mezo.cpp
 * @date   17 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the MeZO optimizer.
 */

#include <mezo.h>
#include <node_exporter.h>
#include <random>
#include <util_func.h>

namespace nntrainer {

MeZO::MeZO() : mezo_props(MeZOEpsilon(), MeZOLearningRate()) {

  auto &[epsilon, learningRate] = mezo_props;
  epsilon.set(0.01f);
  learningRate.set(0.0001f);
}

void MeZO::perturbParameters(std::vector<Tensor *> &params, float epsilon,
                             int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  for (auto *ptr : params) {
    Tensor noise_tensor(ptr->getDim());
    noise_tensor.allocate();

    float *noise_data = (float *)(noise_tensor.getData());
    size_t param_size = ptr->getDim().getDataLen();
    for (size_t i = 0; i < param_size; ++i) {
      noise_data[i] = normal_dist(gen);
    }
    // θi ← θi + ϵz
    ptr->add_i(noise_tensor, epsilon);
  }
}

void MeZO::trainStep(std::function<void()> forward_fn,
                     std::function<float()> get_loss_fn,
                     std::vector<Tensor *> &params) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distrib(0, 1000000000 - 1);
  int seed = distrib(gen);

  float mezo_epsilon = getEpsilon();
  float lr = getLearningRate();

  // Perturb parameters with +epsilon
  perturbParameters(params, mezo_epsilon, seed);

  // Forward pass with positive perturbation
  forward_fn();
  float loss_plus = get_loss_fn();

  // Perturb parameters with -2*epsilon (to get from +epsilon to -epsilon)
  perturbParameters(params, -2 * mezo_epsilon, seed);

  // Forward pass with negative perturbation
  forward_fn();
  float loss_minus = get_loss_fn();

  // Restoring weight to original state
  perturbParameters(params, mezo_epsilon, seed);

  float projected_grad = (loss_plus - loss_minus) / (2.0f * mezo_epsilon);

  updateWeightsMeZO(params, seed, projected_grad);
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
                             int seed, float projected_grad) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  float mezo_epsilon = getEpsilon();
  float lr = getLearningRate();

  for (auto *ptr : weights) {
    Tensor noise_tensor(ptr->getDim());
    noise_tensor.allocate();

    float *noise_data = (float *)(noise_tensor.getData());
    size_t param_size = ptr->getDim().getDataLen();
    for (size_t i = 0; i < param_size; ++i) {
      noise_data[i] = normal_dist(gen);
    }
    // θi ← θi −ηt ∗ projected_grad ∗ z
    ptr->add_i(noise_tensor, -lr * projected_grad);
  }
}

} // namespace nntrainer
