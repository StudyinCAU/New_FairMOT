cd src
nohup python train.py mot --exp_id bdd100k_hrnet_18 --arch 'hrnet_18' --data_cfg '../src/lib/cfg/bdd100k.json' \
    --gpus '0' --num_workers 16 --seed 3047 --batch_size 8 --lr_step '7, 9' \
    --num_epochs 10 --lr 1e-4 --master_batch_size -1 --val_intervals 1 \
    --load_model '/home/station1/weidongtang/report/work2/FairMOT/exp/mot/bdd100k_hrnet_18/model_last.pth' \
    --resume >> ../nohup_log/nohup_bdd100k_hrnet_18.out 2>&1 &
cd ..