mkdir -p logs/qualcomm-baselines
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm random --r_target 20 --n_clusters 10 2>&1 | tee logs/qualcomm-baselines/random_20_10.log
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm random --r_target 20 --n_clusters 100 2>&1 | tee logs/qualcomm-baselines/random_20_100.log
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm random --r_target 20 --n_clusters 1000 2>&1 | tee logs/qualcomm-baselines/random_20_1000.log

python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm kmeans --r_target 20 --n_clusters 10 2>&1 | tee logs/qualcomm-baselines/kmeans_20_10.log
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm kmeans --r_target 20 --n_clusters 100 2>&1 | tee logs/qualcomm-baselines/kmeans_20_100.log
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm kmeans --r_target 20 --n_clusters 1000 2>&1 | tee logs/qualcomm-baselines/kmeans_20_1000.log
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm hdbscan --r_target 20 2>&1 | tee logs/qualcomm-baselines/hdbscan_20.log