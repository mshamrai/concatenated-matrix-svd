# qualcomm experiments
# mkdir -p logs/qualcomm-scatter-rank
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 5 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/5.log
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 10 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/10.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 15 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/15.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/20.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 25 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/25.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 30 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/30.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 35 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/35.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 40 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/40.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 45 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/45.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 50 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-rank/50.log


# mkdir -p logs/qualcomm-scatter-eps
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.01 2>&1 | tee logs/qualcomm-scatter-eps/1.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.02 2>&1 | tee logs/qualcomm-scatter-eps/2.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.03 2>&1 | tee logs/qualcomm-scatter-eps/3.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.04 2>&1 | tee logs/qualcomm-scatter-eps/4.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm-scatter-eps/5.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.06 2>&1 | tee logs/qualcomm-scatter-eps/6.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.07 2>&1 | tee logs/qualcomm-scatter-eps/7.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.08 2>&1 | tee logs/qualcomm-scatter-eps/8.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.09 2>&1 | tee logs/qualcomm-scatter-eps/9.log
# python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.1 2>&1 | tee logs/qualcomm-scatter-eps/10.log

# # smolvlm experiments
# mkdir -p logs/smolvlm-scatter-rank
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 2 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/2.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 4 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/4.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 8 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/8.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 16 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/16.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/32.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 64 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/64.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 128 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/128.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 256 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/256.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 512 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/512.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 1024 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-rank/1024.log


# mkdir -p logs/smolvlm-scatter-eps
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.05 2>&1 | tee logs/smolvlm-scatter-eps/5.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.1 2>&1 | tee logs/smolvlm-scatter-eps/10.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.15 2>&1 | tee logs/smolvlm-scatter-eps/15.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm-scatter-eps/20.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.25 2>&1 | tee logs/smolvlm-scatter-eps/25.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.3 2>&1 | tee logs/smolvlm-scatter-eps/30.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.35 2>&1 | tee logs/smolvlm-scatter-eps/35.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.4 2>&1 | tee logs/smolvlm-scatter-eps/40.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.45 2>&1 | tee logs/smolvlm-scatter-eps/45.log
# python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.5 2>&1 | tee logs/smolvlm-scatter-eps/50.log