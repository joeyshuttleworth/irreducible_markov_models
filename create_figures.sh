#!/usr/bin/bash

set -e

git submodule update --remote --init
mkdir -p paper_figures

# Check Julia
if ! command -v julia >/dev/null 2>&1; then
    echo "Julia is not installed."
    exit 1
fi

# Create environment if it doesn't exist
ENV_NAME="juliaenv"

rm -rf .tmp_venv
python3 -m venv .tmp_venv
. .tmp_venv/bin/activate

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env create -n ${ENV_NAME} -f "environment.yml" -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

PYTHON=$(which python)

julia -e '
ENV["PYTHON"] = "'"$PYTHON"'"
using Pkg

Pkg.add("BifurcationKit")
Pkg.add("DifferentialEquations")
Pkg.add("ArgParse")
Pkg.add("ForwardDiff")
Pkg.build("PyCall")

Pkg.build("PyCall")
'

# Install Python package
julia scripts/bifurcation_plot.jl --output paper_output

pip install --upgrade pip
pip install -r requirements.txt

python3 scripts/degenerate_model_demonstration.py -o paper_output
python3 scripts/compare_model_reductions.py -o paper_output
python3 scripts/simple_drug_model_sim.py -o paper_output

cp paper_output/simple_drug_model/drug_model_output.pdf paper_figures/Fig4.pdf
cp paper_output/compare_model_reductions/compare_reductions.pdf paper_figures/Fig5.pdf
cp paper_output/degenerate_model_demo/degenerate_model_fig.pdf paper_figures/Fig7.pdf
cp paper_output/bifurcation_plot/bifurcation_diagram.pdf paper_figures/sup_Fig8.pdf

deactivate
rm -rf .tmp_venv
