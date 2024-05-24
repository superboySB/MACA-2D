# MACA-2D

MACA-2D is a multi-agent air combat secnario, based on [MACA](https://github.com/CETC-TFAI/MaCA).

<img src="https://simsimi.oss-cn-beijing.aliyuncs.com/test.gif" alt="test" style="zoom:50%;" />

## Quick Start

#### Install

```shell
conda create -n maca
conda install requests[socks]
pip install -r requirements.txt
```

#### Train & Test

```shell
# demo
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python3 demo.py

# train with rllib
python3 train_cr_rllib.py
# test with rllib
python3 test_cr_rllib.py
```

