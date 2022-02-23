python main.py --default_root_dir="experiments/weighted_frames" --gpus=1 --event_representation="weighted_frames" --timesteps=20 --dataset="n-mnist" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_averaging" --gpus=1 --event_representation="weighted_frames" --blur_type="averaging" --timesteps=20 --dataset="n-mnist" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_gaussian" --gpus=1 --event_representation="weighted_frames" --blur_type="gaussian" --timesteps=20 --dataset="n-mnist" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_median" --gpus=1 --event_representation="weighted_frames" --blur_type="median" --timesteps=20 --dataset="n-mnist" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/bit_encoding" --gpus=1 --event_representation="bit_encoding" --timesteps=8 --dataset="n-mnist" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001



python main.py --default_root_dir="experiments/weighted_frames" --gpus=1 --event_representation="weighted_frames" --timesteps=20 --dataset="cifar10-dvs" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_averaging" --gpus=1 --event_representation="weighted_frames" --blur_type="averaging" --timesteps=20 --dataset="cifar10-dvs" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_gaussian" --gpus=1 --event_representation="weighted_frames" --blur_type="gaussian" --timesteps=20 --dataset="cifar10-dvs" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_median" --gpus=1 --event_representation="weighted_frames" --blur_type="median" --timesteps=20 --dataset="cifar10-dvs" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/bit_encoding" --gpus=1 --event_representation="bit_encoding" --timesteps=8 --dataset="cifar10-dvs" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001


python main.py --default_root_dir="experiments/weighted_frames" --gpus=1 --event_representation="weighted_frames" --timesteps=20 --dataset="dvsgesture" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_averaging" --gpus=1 --event_representation="weighted_frames" --blur_type="averaging" --timesteps=20 --dataset="dvsgesture" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_gaussian" --gpus=1 --event_representation="weighted_frames" --blur_type="gaussian" --timesteps=20 --dataset="dvsgesture" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/weighted_frames_median" --gpus=1 --event_representation="weighted_frames" --blur_type="median" --timesteps=20 --dataset="dvsgesture" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001

python main.py --default_root_dir="experiments/bit_encoding" --gpus=1 --event_representation="bit_encoding" --timesteps=8 --dataset="dvsgesture" --batch_size=64 --max_epochs=150 --height=224 --width=224 --learning_rate=0.001
@REM 99.4
python main.py --default_root_dir="experiments/voxel_nmnist" --num_workers=8 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.009
python main.py --default_root_dir="experiments/bit_nmnist" --num_workers=8 --gpus=1 --event_representation="bit_encoding" --timesteps=8 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.0019
@REM 99.4
python main.py --default_root_dir="experiments/framestime_nmnist" --num_workers=8 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.0019 --mode="lr_find"

python main.py --default_root_dir="experiments/framestime_nmnist" --num_workers=8 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.0019 --mode="lr_find"


python main.py --default_root_dir="experiments/framestime_cifar10" --num_workers=4 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
python main.py --default_root_dir="experiments/voxel_cifar10" --num_workers=4 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
python main.py --default_root_dir="experiments/histogram_cifar10" --num_workers=4 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
python main.py --default_root_dir="experiments/bit_cifar10" --num_workers=4 --gpus=1 --event_representation="bit_encoding" --timesteps=8 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1
python main.py --default_root_dir="experiments/hots_cifar10" --num_workers=4 --gpus=1 --event_representation="HOTS" --timesteps=8 --dataset="cifar10-dvs" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1

python main.py --default_root_dir="experiments/hots_nmnist" --num_workers=4 --gpus=1 --event_representation="HOTS" --timesteps=8 --dataset="n-mnist" --batch_size=1024 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1


@REM 79.55
python main.py --default_root_dir="experiments/voxel_gesture" --num_workers=2 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 82.95
python main.py --default_root_dir="experiments/framestime_gesture" --num_workers=2 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 90.91
python main.py --default_root_dir="experiments/histogram_gesture" --num_workers=2 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
@REM 80.68
python main.py --default_root_dir="experiments/bit_gesture" --num_workers=2 --gpus=1 --event_representation="bit_encoding" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/HOTS_gesture" --num_workers=2 --gpus=1 --event_representation="hots_encoding" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001 --log_every_n_steps=1



python main.py --default_root_dir="experiments/voxel_gesture" --num_workers=4 --gpus=1 --event_representation="VoxelGrid" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/framestime_gesture" --num_workers=4 --gpus=1 --event_representation="frames_time" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/histogram_gesture" --num_workers=4 --gpus=1 --event_representation="histogram" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001
python main.py --default_root_dir="experiments/bit_gesture" --num_workers=4 --gpus=1 --event_representation="bit_encoding" --timesteps=10 --dataset="dvsgesture" --batch_size=256 --max_epochs=60 --height=224 --width=224 --learning_rate=0.001