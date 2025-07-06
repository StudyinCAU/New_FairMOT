export PYTHONWARNINGS="ignore"
cd src
nohup python train.py mot --exp_id bdd100k_ddrnet_23 --arch 'ddrnet_23' --data_cfg '../src/lib/cfg/bdd100k.json' \
    --gpus '0' --num_workers 16 --seed 3047 --batch_size 8 --lr_step '7, 9' --print_iter 100 \
    --num_epochs 2 --lr 1e-4 --master_batch_size -1 --val_intervals 1 \
    > ../nohup_log/nohup_bdd100k_ddrnet_23.out 2>&1 &
cd ..