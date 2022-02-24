
@REM 99.43
python main.py --default_root_dir="experiments/voxel_nmnist" --num_workers=8 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.009
@REM 99.34
python main.py --default_root_dir="experiments/bit_nmnist" --num_workers=8 --gpus=1 --event_representation="bit_encoding" --timesteps=1 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.0019
@REM 99.52
python main.py --default_root_dir="experiments/multibit_nmnist" --num_workers=10 --gpus=1 --event_representation="bit_encoding" --timesteps=3 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 99.36
python main.py --default_root_dir="experiments/framestime_nmnist" --num_workers=8 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.0019 --mode="lr_find"
@REM 99.14
python main.py --default_root_dir="experiments/histo_nmnist" --num_workers=8 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.0019 --mode="lr_find"

@REM 93.45
python main.py --default_root_dir="experiments/framestime_cifar10" --num_workers=0 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 94.00
python main.py --default_root_dir="experiments/voxel_cifar10" --num_workers=0 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 93.65
python main.py --default_root_dir="experiments/histogram_cifar10" --num_workers=0 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 92.4
python main.py --default_root_dir="experiments/bit_cifar10" --num_workers=0 --gpus=1 --event_representation="bit_encoding" --timesteps=1 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 93.25
python main.py --default_root_dir="experiments/multibit_cifar10" --num_workers=10 --gpus=1 --event_representation="bit_encoding" --timesteps=3 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1



@REM 92.48
python main.py --default_root_dir="experiments/framestime_ncars" --num_workers=10 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="ncars" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 90.71
python main.py --default_root_dir="experiments/voxel_ncars" --num_workers=10 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="ncars" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 91.77
python main.py --default_root_dir="experiments/histogram_ncars" --num_workers=10 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="ncars" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
@REM 92.04
python main.py --default_root_dir="experiments/bit_ncars" --num_workers=10 --gpus=1 --event_representation="bit_encoding" --timesteps=1 --dataset="ncars" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1

python main.py --default_root_dir="experiments/multibit_ncars" --num_workers=10 --gpus=1 --event_representation="bit_encoding" --timesteps=3 --dataset="ncars" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1




python main.py --default_root_dir="experiments/hots_cifar10" --num_workers=10 --gpus=1 --event_representation="HOTS" --timesteps=8 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
python main.py --default_root_dir="experiments/hots_nmnist" --num_workers=5 --gpus=1 --event_representation="HOTS" --timesteps=8 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1


@REM 79.55
python main.py --default_root_dir="experiments/voxel_gesture" --num_workers=2 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 82.95
python main.py --default_root_dir="experiments/framestime_gesture" --num_workers=2 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 90.91
python main.py --default_root_dir="experiments/histogram_gesture" --num_workers=2 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 80.68
python main.py --default_root_dir="experiments/bit_gesture" --num_workers=2 --gpus=1 --event_representation="bit_encoding" --timesteps=1 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 87.88
python main.py --default_root_dir="experiments/multibit_gesture" --num_workers=10 --gpus=1 --event_representation="bit_encoding" --timesteps=5 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/HOTS_gesture" --num_workers=2 --gpus=1 --event_representation="hots_encoding" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1



python main.py --default_root_dir="experiments/voxel_gesture" --num_workers=4 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/framestime_gesture" --num_workers=4 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/histogram_gesture" --num_workers=4 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/bit_gesture" --num_workers=4 --gpus=1 --event_representation="bit_encoding" --timesteps=1 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001