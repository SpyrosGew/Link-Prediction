# Link-Prediction
Link Prediction is a university Project for the course of Graph Technologies

## Create python 3.10 environment

you can do this either with vevn or conda

### venv 

```bash
sudo apt install python3.10 python3.10-venv
```

Navigate inside the project and create the environment

```bash
cd Link-Prediction
python3.10 -m venv .venv
```

Activate the environment

```bash
source .venv/bin/activate
#check with
python --version
```

### conda

Create the environment and activate it

```bash
conda create -n linkpred_env python=3.10
conda activate linkpred_env
```

## Install requirments

Inside the project directory 

```bash
pip install -r requirements.txt
```
