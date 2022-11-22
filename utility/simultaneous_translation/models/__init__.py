# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

ignores = ["simple_nat.py", "toy_transformer.py"]
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_") and file not in ignores:
        model_name = file[: file.find(".py")]
        importlib.import_module(
            "simultaneous_translation.models." + model_name
        )
