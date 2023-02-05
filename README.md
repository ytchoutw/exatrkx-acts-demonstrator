# Exa.TrkX ACTS demonstrator

Small script that runs the Exa.TrkX track finding module in ACTS with data simulated in the OpenDataDetector with the FATRAS fast simulation.

## Usage

Build ACTS with e.g.

```bash
cmake .. \
    -D ACTS_BUILD_PLUGIN_DD4HEP=ON \
    -D ACTS_BUILD_PLUGIN_EXATRKX=ON \
    -D ACTS_BUILD_FATRAS=ON \
    -D ACTS_BUILD_EXAMPLES=ON \
    -D ACTS_BUILD_EXAMPLES_DD4HEP=ON \
    -D ACTS_BUILD_EXAMPLES_EXATRKX=ON \
    -D ACTS_BUILD_EXAMPLES_PYTHIA8=ON \
    -D ACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON \
    -D ACTS_BUILD_ODD=ON
```

Setup the environment, e.g. with a bash script `run_inference.sh` like:

```bash
#!/bin/bash

source <path-to-acts>/build/python/setup.sh
source <path-to-dd4hep>/bin/thisdd4hep_only.sh
export LD_LIBRARY_PATH=<path-to-acts>/build/thirdparty/OpenDataDetector/factory:$LD_LIBRARY_PATH

python3 inference.py "$@"
```

The script requires some options:

```bash
./run_inference.sh <n_events> <path-to-models> <digi-config>
```

The repository contains two sets of models, one trained with *truth hits* and one with *smeared digitization*. Along with this, pass `truth` or `smear` as *digitization config*, this loads the corresponding digitization config file.

The small evaulation script `evaluate.py` takes the resulting `track_finding_performance_exatrkx.root`, and makes a simple performance plot.

## Requirements

* recent ACTS version + ODD + Exa.TrkX Plugin
* DD4hep
* Pythia8
* CUDA + cuDNN
* libtorch

