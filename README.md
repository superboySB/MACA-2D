# MACA-2D

MACA-2D is a multi-agent air combat secnario, based on [MACA](https://github.com/CETC-TFAI/MaCA).

<img src="https://simsimi.oss-cn-beijing.aliyuncs.com/test.gif" alt="test" style="zoom:50%;" />

## Quick Start

#### Install

```shell
conda create -n maca
conda install requests[socks] tensorboard
pip install -r requirements.txt
```
如果提示`GLIBCXX_3.4.30’ not found 的情况，则
```sh
cd /home/$USER/miniconda3/maca/lib
mkdir backup  # Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.29
```

#### Train & Test

```shell
# demo
python3 demo.py

# train with rllib
python3 train_cr_rllib.py
# test with rllib
python3 test_cr_rllib.py
```

尝试在洪都杯的环境上直接运行训练
```sh
D:\projects\hdbisai\python.exe demo.py
```

