# Efficient Reinforcement Learning Library (ERLlib)

A deep reinforcement learning library that provides efficient implementation with regard
to sample and computation.

## Algorithms

- SAC
- ESVG

## Setup execution environment

At first, install suitable version of pytorch<2, then

```shell
pip install -e .
```

## How to reproduce experiment

Execute ESVG algorithm with option `agent=svg`

```shell 
python train.py agent=svg log.log_dir=/tmp/svg env=dmc env.task_id=cheetah-run
```

for the Gym tasks, set `env=gym` and `env.task_id` to one of "MBHumanoid-v0", "
MBAnt-v0", "MBHopper-v0", "MBHalfCheetah-v0" and "MBWalker2d-v0".
