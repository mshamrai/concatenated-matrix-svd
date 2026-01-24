# qualcomm experiments
mkdir -p logs/qualcomm
echo "Running qualcomm experiments..."
echo "Running max_norm..."
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm max_norm --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm/max_norm.log
echo "Running residuals with norm sorting..."
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm residuals --sorting_strategy norm --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm/residuals_norm.log
echo "Running residuals with residual sorting..."
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm residuals --sorting_strategy residual --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm/residuals_residual.log
echo "Running approximate with norm sorting..."
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy norm --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm/approximate_norm.log
echo "Running approximate with residual sorting..."
python -m src.main --dataset qualcomm --data_path "data/qdndata_batch17/QDNData" --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.05 2>&1 | tee logs/qualcomm/approximate_residual.log


# # bigearth experiments
mkdir -p logs/bigearth
echo "Running bigearth experiments..."
echo "Running max_norm..."
python -m src.main --dataset bigearth --data_path "data/BigEarthNet-S1" --max_samples 10000 --algorithm max_norm --r_target 20 --eps 0.05 --random_seed 23 2>&1 | tee logs/bigearth/max_norm.log
echo "Running residuals with norm sorting..."
python -m src.main --dataset bigearth --data_path "data/BigEarthNet-S1" --max_samples 10000 --algorithm residuals --sorting_strategy norm --r_target 20 --eps 0.05 --random_seed 23 2>&1 | tee logs/bigearth/residuals_norm.log
echo "Running residuals with residual sorting..."
python -m src.main --dataset bigearth --data_path "data/BigEarthNet-S1" --max_samples 10000 --algorithm residuals --sorting_strategy residual --r_target 20 --eps 0.05 --random_seed 23 2>&1 | tee logs/bigearth/residuals_residual.log
echo "Running approximate with norm sorting..."
python -m src.main --dataset bigearth --data_path "data/BigEarthNet-S1" --max_samples 10000 --algorithm approximate --sorting_strategy norm --r_target 20 --eps 0.05 --random_seed 23 2>&1 | tee logs/bigearth/approximate_norm.log
echo "Running approximate with residual sorting..."
python -m src.main --dataset bigearth --data_path "data/BigEarthNet-S1" --max_samples 10000 --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.05 --random_seed 23 2>&1 | tee logs/bigearth/approximate_residual.log


# # pdebench experiments
mkdir -p logs/pdebench
echo "Running pdebench experiments..."
echo "Running max_norm..."
python -m src.main --dataset pdebench --data_path "data/pdebench/1D/Advection/Train" --max_samples 5000 --algorithm max_norm --r_target 67 --eps 0.05 2>&1 | tee logs/pdebench/max_norm.log
echo "Running approximate with norm sorting..."
python -m src.main --dataset pdebench --data_path "data/pdebench/1D/Advection/Train" --max_samples 5000 --algorithm approximate --sorting_strategy norm --r_target 67 --eps 0.05 2>&1 | tee logs/pdebench/approximate_norm.log
echo "Running approximate with residual sorting..."
python -m src.main --dataset pdebench --data_path "data/pdebench/1D/Advection/Train" --max_samples 5000 --algorithm approximate --sorting_strategy residual --r_target 67 --eps 0.05 2>&1 | tee logs/pdebench/approximate_residual.log
echo "Running residuals with norm sorting..."
python -m src.main --dataset pdebench --data_path "data/pdebench/1D/Advection/Train" --max_samples 5000 --algorithm residuals --sorting_strategy norm --r_target 67 --eps 0.05 2>&1 | tee logs/pdebench/residuals_norm.log
echo "Running residuals with residual sorting..."
python -m src.main --dataset pdebench --data_path "data/pdebench/1D/Advection/Train" --max_samples 5000 --algorithm residuals --sorting_strategy residual --r_target 67 --eps 0.05 2>&1 | tee logs/pdebench/residuals_residual.log


# smolvlm experiments
mkdir -p logs/smolvlm
echo "Running smolvlm experiments..."
echo "Running max_norm..."
python -m src.main --dataset smolvlm --algorithm max_norm --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm/max_norm.log
echo "Running residuals with norm sorting..."
python -m src.main --dataset smolvlm --algorithm residuals --sorting_strategy norm --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm/residuals_norm.log
echo "Running residuals with residual sorting..."
python -m src.main --dataset smolvlm --algorithm residuals --sorting_strategy residual --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm/residuals_residual.log
echo "Running approximate with norm sorting..."
python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy norm --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm/approximate_norm.log
echo "Running approximate with residual sorting..."
python -m src.main --dataset smolvlm --algorithm approximate --sorting_strategy residual --r_target 32 --eps 0.2 2>&1 | tee logs/smolvlm/approximate_residual.log