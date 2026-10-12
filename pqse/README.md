# Power Quality State Estimation
State estimation algorithms for power quality data.
- DNN: Dense Neural Network
- PANN: Physics-Aware Neural Network
- PINN: Physics-Informed Neural Network (Work in Progress)

## Install
Install requirements from `requirements.txt`:
```bash
pip install -r requirements.txt
```
Environment variable `LD_LIBRARY_PATH` must be set to include the CUDA libraries (Linux Ubuntu):
`LD_LIBRARY_PATH=/lib:/home/sim/miniconda3/envs/tf/lib/:/home/sim/miniconda3/envs/tf/lib/python3.9/site-packages/nvidia/cudnn/lib:/home/sim/miniconda3/envs/tf/lib/:;PYTHONUNBUFFERED=1;XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda`

## Run
1. Configure database connection in `config.ini`.
2. Edit the hyperparameters in the main file, here you can also chose a model
3. Run the main file to start the state estimation.
