set -e

python src/data_refresh.py
python src/inference.py
