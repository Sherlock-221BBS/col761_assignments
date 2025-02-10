eval "$(conda shell.bash hook)"

conda activate whocares


python3 code.py findDiscriminativeSubgraphs \
    --graphs $1\
    --labels $2 \
    --output $3
