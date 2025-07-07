export PYTHONWARNINGS="ignore"
cd src
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python3 track.py mot \
    --exp_id test_bdd100k_vit_b16 \
    --arch vit_16 --load_model ../exp/mot/bdd100k_vit_b16/model_last.pth \
    --gpus '0' --conf_thres 0.4 --data_cfg ../src/lib/cfg/bdd100k.json \
    --data_dir ../dataset/bdd100kmot_vehicle/images/val \
    --num_workers 16 --batch_size 8 --seed 3047" \
    >> ../nohup_log/nohup_test_bdd100k_vit_b16.out 2>&1 &
cd ..