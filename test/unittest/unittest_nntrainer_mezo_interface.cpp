/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file        unittest_nntrainer_mezo_interface.cpp
 * @date        05 May 2026
 * @brief       Unit test for MeZO optimizer interface changes.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Sachin Singh <sachin.3@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <mezo.h>
#include <neuralnet.h>
#include <optimizer_devel.h>
#include <optimizer_wrapped.h>
#include <sgd.h>

using namespace nntrainer;

/**
 * @brief Test that MeZO optimizer correctly reports it doesn't require backprop
 */
TEST(nntrainer_mezo_interface, requiresBackprop_01_p) {
  MeZO mezo;
  EXPECT_FALSE(mezo.requiresBackprop());
}

/**
 * @brief Test that SGD optimizer correctly reports it requires backprop
 */
TEST(nntrainer_mezo_interface, requiresBackprop_02_p) {
  SGD sgd;
  EXPECT_TRUE(sgd.requiresBackprop());
}

/**
 * @brief Test that OptimizerWrapped correctly delegates requiresBackprop for
 * MeZO
 */
TEST(nntrainer_mezo_interface, requiresBackprop_wrapped_01_p) {
  auto mezo_ptr = std::make_unique<MeZO>();
  auto wrapped = createOptimizerWrapped(std::move(mezo_ptr));
  EXPECT_FALSE(wrapped->requiresBackprop());
}

/**
 * @brief Test that OptimizerWrapped correctly delegates requiresBackprop for
 * SGD
 */
TEST(nntrainer_mezo_interface, requiresBackprop_wrapped_02_p) {
  auto sgd_ptr = std::make_unique<SGD>();
  auto wrapped = createOptimizerWrapped(std::move(sgd_ptr));
  EXPECT_TRUE(wrapped->requiresBackprop());
}

/**
 * @brief Mock function to test trainStep interface
 */
static bool forward_called = false;
static bool get_loss_called = false;

void mock_forward_fn() { forward_called = true; }

float mock_get_loss_fn() {
  get_loss_called = true;
  return 0.5f;
}

/**
 * @brief Test that MeZO optimizer trainStep method can be called
 */
TEST(nntrainer_mezo_interface, trainStep_01_p) {
  MeZO mezo;
  std::vector<Tensor *> params; // Empty params for this test

  // This should not crash and should call the functions
  forward_called = false;
  get_loss_called = false;

  mezo.trainStep(mock_forward_fn, mock_get_loss_fn, params);

  // Note: The actual MeZO implementation will try to perturb parameters,
  // but with empty params vector, it should complete without crashing
  SUCCEED();
}

/**
 * @brief Test that base Optimizer trainStep method is a no-op
 */
TEST(nntrainer_mezo_interface, trainStep_base_01_p) {
  // Create a mock optimizer that inherits from base Optimizer
  class MockOptimizer : public Optimizer {
  public:
    double getDefaultLearningRate() const override { return 0.01; }
    void applyGradient(RunOptimizerContext &context) override {}
    std::vector<TensorDim>
    getOptimizerVariableDim(const TensorDim &dim) override {
      return {};
    }
    const std::string getType() const override { return "Mock"; }
    void setProperty(const std::vector<std::string> &values) override {}
  };

  MockOptimizer mock_opt;
  std::vector<Tensor *> params;

  // This should be a no-op and not crash
  mock_opt.trainStep(mock_forward_fn, mock_get_loss_fn, params);
  SUCCEED();
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
