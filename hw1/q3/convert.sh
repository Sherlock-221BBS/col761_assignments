eval "$(conda shell.bash hook)"

conda activate whocares

python3 code.py convertToFeature \
    --graphs $1 \
    --subgraphs $2 \
    --output $3
