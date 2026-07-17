# Irreducible Markov Models

Repository for paper on Irreducible Markov Models in Mathematical Biology.

## Getting Started

### Dependencies

* Python >= 3.11
* Pip
* [Myokit](https://myokit.org)
* A Julia installation

### Installing

Ensure pip is up-to-date:
```python3 -m pip install --upgrade pip```

Install required packages from `requirements.txt`:
```
    python3 -m pip install -r requirements.txt
```

You should now be able to run the code.

### Executing program

Open jupyter by running
```
python3 -m jupyter-notebook notebooks/model_reduction_demonstration.ipynb
```

You should now be able to run the code in the notebook. All Figures for the paper can be created by running the bash script from this root directory like so:

```
./create_figures.sh
```
