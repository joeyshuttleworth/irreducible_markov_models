#!/usr/bin/bash

mkdir -p paper_figures

python3 -m venv .tmp_venv
. .tmp_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 scripts/degenerate_model_demonstration.py -o paper_output
python3 scripts/compare_model_reductions.py -o paper_output
julia scripts/bifurcation_plot.jl --output paper_output
python3 scripts/simple_drug_model_sim.py -o paper_output

cp paper_output/simple_drug_model/drug_model_output.pdf paper_figures/Fig5.pdf
cp paper_output/compare_model_reductions/compare_reductions.pdf paper_figures/Fig6.pdf
cp paper_output/degenerate_model_demo/degenerate_model_fig.pdf paper_figures/Fig7.pdf
cp paper_output/bifurcation_plot/bifurcation_diagram.pdf paper_figures/Fig8.pdf

deactivate
rm -rf .tmp_venv
