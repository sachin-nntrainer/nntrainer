// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   26 Feb 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.honge@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>
#ifdef ENABLE_ONNX_INTERPRETER
#include <onnx_interpreter.h>
#include <unordered_map>
#endif

int main() {
  auto model = ml::train::createModel();

  try {
    std::string path = "../../../../Applications/ONNX/jni/"
                       "fcModel/matmul_dynamic.onnx";

    nntrainer::ONNXInterpreter interp;

    // Example user-provided mapping of symbolic dimension names to values.
    std::unordered_map<std::string, int> dim_map = {{"seq", 7}};

    // This step should be done before 'deserialize'
    interp.setDimParamMap(dim_map);

    auto graph = interp.deserialize(path);

    // add layers produced by the interpreter into the model
    for (auto &node : graph) {
      model->addLayer(node);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  // NOTE: Havent checked inference but this PR contains weight loading commits.
  // One can try. Ref: PR #3648

  return 0;
}
