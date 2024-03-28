Check out `class.py` there's a bunch of different ways of calling the fused CPU optimizers and the techniques work for adam, adamw and a few more popular optimizers

Ideally we want to just upstream this stuff in core

For now we're seeing 10% speedups up to close to 2x so working with Intel on figuring this stuff out


Installation: follow instructions here https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2bcpu&os=linux%2fwsl2&package=pip
