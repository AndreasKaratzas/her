
# Odysseus: DDPG with Hindsight Experience Replay

### Abstract

This is a pytorch implementation of [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). 

### Introduction


### Installation

Linux: https://mpi4py.readthedocs.io/en/stable/install.html

https://neptune.ai/blog/installing-mujoco-to-work-with-openai-gym-environments

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/andreas/.mujoco/mjpro150/bin

sudo apt-get install libosmesa6-dev

sudo apt-get install patchelf


### Usage

1. train the **FetchReach-v1**:
```bash
mpirun -np 1 python main.py --env='FetchReach-v1' --clip-return --device='cuda' --debug-mode --name='FetchReach-v1' --auto-save --logger-name='FetchReach-v1' --checkpoint-dir '../../../../data/experiments'
```
2. train the **FetchPush-v1**:
```bash
mpirun -np 8 python main.py --env='FetchPush-v1' --clip-return --device='cpu' --debug-mode --name='FetchPush-v1' --auto-save --logger-name='FetchPush-v1' --checkpoint-dir '../../../../data/experiments'
```
3. train the **FetchPickAndPlace-v1**:
```bash
mpirun -np 16 python main.py --env='FetchPickAndPlace-v1' --clip-return --device='cpu' --debug-mode --name='FetchPickAndPlace-v1' --auto-save  --logger-name='FetchPickAndPlace-v1' --checkpoint-dir '../../../../data/experiments'
```
4. train the **FetchSlide-v1**:
```bash
mpirun -np 1 python main.py --env='FetchSlide-v1' --clip-return --device='cuda' --debug-mode --name='FetchSlide-v1' --auto-save --logger-name='FetchSlide-v1' --checkpoint-dir '../../../../data/experiments'
```

### Demo
```bash
```

### Experiments

Table with pretrained model stats

### Acknowledgement

- [Openai Baselines](https://github.com/openai/baselines)
- [TianhongDai](https://github.com/TianhongDai/hindsight-experience-replay)

### Future Work

- [ ] Add `tensorboard`
- [ ] Add `docstrings`
- [ ] Add instructions for mujoco on Windows
- [ ] Experiments
- [ ] Include pretrained models
- [ ] Include examples
- [ ] Complete `README`
