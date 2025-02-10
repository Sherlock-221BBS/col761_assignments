eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "whocares"; then
    echo "Creating Conda environment 'whocares'..."
    conda create -n whocares python=3.10 -y
else
    echo "Conda environment 'whocares' already exists. Skipping creation."
fi

conda activate whocares

pip install numpy scikit-learn networkx==2.8.8

chmod +x gaston
chmod +x convert.sh
chmod +x identify.sh
