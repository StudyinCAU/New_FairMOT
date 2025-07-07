export PYTHONWARNINGS="ignore"
cd src
nohup python3 track.py mot --exp_id test_bdd100k_hrnet_18 \
    --arch hrnet_18 --load_model /home/station1/weidongtang/report/work2/FairMOT/exp/mot/bdd100k_hrnet_18/model_10.pth \
    --gpus 0 --conf_thres 0.4 --data_cfg ../src/lib/cfg/bdd100k.json \
    --data_dir /home/station1/weidongtang/report/work2/FairMOT/dataset/bdd100kmot_vehicle/images/val \
    --num_workers 16 --batch_size 32 --seed 3047 \
    >> ../nohup_log/nohup_test_bdd100k_hrnet_18.out 2>&1 &
cd ..