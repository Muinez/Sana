# Train small model

rent A40 on runpod, cuda 12.1 Torch 2.2.1 100/50Gb space
```
cd /home
apt update
git clone git@github.com:recoilme/Sana.git
cd Sana
pip install -U pip
pip install -e .
pip install -U "huggingface_hub[cli]"
git config --global credential.helper store
huggingface-cli login

# dataset
cd /workspace
wget https://huggingface.co/datasets/recoilme/ae/resolve/main/anime.zip
apt install unzip
unzip -q anime.zip
rm anime.zip
python /home/Sana/train_scripts/make_buckets.py --config=/home/Sana/configs/sana_config/512ms/Potato_600M_img512.yaml --data.data_dir=[/workspace/anime] --data.buckets_file=/workspace/anime.json

# train new, potato model in bf16
torchrun --nproc_per_node=1 /home/Sana/train_scripts/train_local.py --config_path=/home/Sana/configs/sana_config/512ms/Potato_600M_img512.yaml --data.buckets_file=/workspace/anime.json --name=2

# check result: https://wandb.ai/recoilme/potato/runs/2?nw=nwuserrecoilme
# Feel the horror
# save cp
cp potato/checkpoints_last/epoch_60_step_1741.pth /workspace/potato.pth

# continue train if you brave enough
torchrun --nproc_per_node=1 /home/Sana/train_scripts/train_local.py --config_path=/home/Sana/configs/sana_config/512ms/Potato_600M_img512-finetune.yaml --data.buckets_file=/workspace/anime.json --name=3
```