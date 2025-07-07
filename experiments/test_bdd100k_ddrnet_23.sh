export PYTHONWARNINGS="ignore"
cd src
nohup bash -c "CUDA_VISIBLE_DEVICES=4 python3 track.py mot \
    --exp_id test_bdd100k_ddrnet_23 \
    --arch ddrnet_23 --load_model ../exp/mot/bdd100k_ddrnet_23/model_last.pth \
    --gpus '0' --conf_thres 0.4 --data_cfg ../src/lib/cfg/bdd100k.json \
    --data_dir ../dataset/bdd100kmot_vehicle/images/val \
    --num_workers 16 --batch_size 96 --seed 3047" \
    >> ../nohup_log/nohup_test_bdd100k_ddrnet_23.out 2>&1 &
cd ..