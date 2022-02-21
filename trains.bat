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