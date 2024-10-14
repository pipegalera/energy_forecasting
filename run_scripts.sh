set -e

python src/data_refresh.py
python src/data_preprocessing.py
python src/inference.py
