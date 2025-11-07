# FPVNet

## Installation

#### Clone the repository
``` # TODO change this repo to ALRHub's repo
git clone git@github.com:ALRhub/FPVNet.git
git submodule update --init --recursive
```

#### Prerequisites
```
mamba create -n fpv_net python=3.10
mamba activate fpv_net

pip install torch torchvision torchaudio
pip install --no-build-isolation -r requirements.txt
```

#### RoboCasa
```
cd custom_robocasa/
pip install -e custom_robosuite/
pip install -e custom_robocasa/

pip install -r requirements.txt
cd ..

python -m robocasa.scripts.download_kitchen_assets
python -m robosuite.scripts.setup_macros
python -m robocasa.scripts.setup_macros

```

#### SUGAR
```
cd agents/encoders/sugar
pip install -e Pointnet2_PyTorch/pointnet2_ops_lib
# Download pretrained models
wget -O pretrain.tar.gz 'https://www.dropbox.com/scl/fi/wyq9pku4gmpwu2n6en55q/pretrain.tar.gz?rlkey=ma6fyeiittl7bad1ho3vx4qsa&e=1&st=rpc1en7w'
tar -xvf pretrain.tar.gz
rm pretrain.tar.gz
cd ../../..
```

#### Point Cloud Visualizer
```
pip install -e utils/visualizer
```

## Preparing RoboCasa Datasets

#### Download RoboCasa Datasets
```
python -m robocasa.scripts.download_datasets --ds_types human_raw

```

#### Generate Point Cloud Observations
Run the following for each dataset you want to process.

```
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m robocasa.scripts.dataset_states_to_obs \
    --dataset <ds-path> \
    --output processed_demo_128_128.hdf5 \
    --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand \
    --camera_width 128 --camera_height 128 \
    --keep_full_pc --dont_store_image --dont_store_depth
```

## FAQ

#### Setting Dataset Path

Set the value of `DATASET_BASE_PATH` in `custom_robocasa/custom_robocasa/robocasa/macros_private.py` to the path of your dataset.
