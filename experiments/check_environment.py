"""Inspect the interpreter, dependencies, and CUDA availability without training."""
import json
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml


if __name__ == "__main__":
    environment = dict(
        executable=sys.executable, python=platform.python_version(),
        torch=torch.__version__, torchvision=torchvision.__version__,
        numpy=np.__version__, pandas=pd.__version__,
        matplotlib=matplotlib.__version__, pyyaml=yaml.__version__,
        cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        cpu_threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    print(json.dumps(environment, indent=2))
