# Exa.TrkX ACTS demonstrator

Small script that runs the Exa.TrkX track finding module in ACTS with data simulated in the OpenDataDetector with the FATRAS fast simulation.

## Usage

Setup the environment, e.g. with a bash script like:

```bash
source <path-to-acts>/build/python/setup.sh
source <path-to-dd4hep>/thisdd4hep_only.sh
export LD_LIBRARY_PATH=<path-to-acts>/build/thirdparty/OpenDataDetector/factory:$LD_LIBRARY_PATH

python3 inference.py "$@"
```

## Requirements

* recent ACTS version + ODD + Exa.TrkX Plugin
* DD4hep
* Pythia8
* CUDA + cuDNN
* libtorch

