"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>

@file compare.py
@date 17 September 2025
@brief This script compares official and NNTrainer model output logits.
@note This script has been tested with transformers version 4.55.0 and PyTorch version 2.8.0

@author Sachin Singh <sachin.3@samsung.com>
"""

import numpy as np

arr1 = np.fromfile("./modelling_logits.bin", dtype="float32").reshape(1, 151936)
arr2 = np.fromfile("./nntrainer_logits.bin", dtype="float32").reshape(1, 151936)

if np.allclose(arr1, arr2, atol=1e-4, rtol=1e-4):
    print("equal")
else:
    print("not equal")
