export PYTHONWARNINGS="ignore"
cd src

nohup bash -c "CUDA_VISIBLE_DEVICES=4 python train.py mot \
    --exp_id bdd100k_dla_34 \
    --arch 'dla_34' \
    --data_cfg '../src/lib/cfg/bdd100k.json' \
    --gpus '0' \
    --num_workers 16 \
    --seed 3047 \
    --batch_size 48 \
    --lr_step '7,9' \
    --print_iter 100 \
    --num_epochs 2 \
    --lr 1e-4 \
    --master_batch_size -1 \
    --val_intervals 1" \
    > ../nohup_log/nohup_bdd100k_dla_34.out 2>&1 &

cd ..
