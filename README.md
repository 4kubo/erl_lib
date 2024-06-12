# Efficient Reinforcement Learning Library (ERLlib)

A deep reinforcement learning library that provides efficient implementation with regard
to sample and computation.

## Algorithms

- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1812.05905)
- Efficient Stochastic Value Gradient (ESVG)
  by [Akihiro Kubo](https://scholar.google.com/citations?user=qVpVG6gAAAAJ&hl=en), [Paavo Parmas](https://scholar.google.com/citations?user=IXbKCUYAAAAJ&hl=en&oi=sra)
  and [Shin Ishii](https://scholar.google.com/citations?user=b-EbgD4AAAAJ&hl=en&oi=sra):
  Sample efficient and stable implementation
  of Stochastic Value Gradient on top of the SAC algorithm,
  or [SAC-SVG(H)](https://proceedings.mlr.press/v144/amos21a.html)

## Setup execution environment

At first, install suitable version of pytorch<2, then

```shell
pip install -e .
```

## How to reproduce experiment

Execute ESVG algorithm with option `agent=svg`

```shell 
python train.py agent=svg env=dmc env.task_id=cheetah-run
```

for the [Gymnaisum](https://gymnasium.farama.org/) tasks, set `env=gym`
and `env.task_id` to one of "MBHumanoid-v0", "
MBAnt-v0", "MBHopper-v0", "MBHalfCheetah-v0" and "MBWalker2d-v0".

Other available value for `env` is `dmc`
for [DM control](https://github.com/google-deepmind/dm_control), `gym-robo`
for [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
and `myosuite` for [MyoSuite](https://github.com/MyoHub/myosuite).