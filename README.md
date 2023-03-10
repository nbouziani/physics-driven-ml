# Physics-driven machine learning models

Developing and running physics-driven machine learning models in a highly productive way using PyTorch and Firedrake


## Table of Contents
* [Setup](#setup)
* [Generate dataset](#generate-dataset)
* [Training](#training)
* [Evaluation](#evaluation)
* [Citation](#citation)

## Setup

This work relies on the Firedrake finite element system, which needs to be installed.

### Installing Firedrake

Firedrake is installed via its installation script, which you can download and run via:

```install_firedrake
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python3 firedrake-install --package-branch firedrake pytorch_coupling
```

Finally, you will need to activate the Firedrake virtual environment:

```activate_venv
source firedrake/bin/activate
```

For more details about installing Firedrake: see [here](https://www.firedrakeproject.org/download.html).

### Testing installation

We recommend that you run the test suite after installation to check that your setup is fully functional. Activate the venv as above and then run:

```install_firedrake_external_operator_branches
pytest tests
```


## Generate dataset

You can generate your own dataset with . This generates pairs $\lbrace\kappa_{i}, u_{i}^{obs}\rbrace_{1 \le i\le n}$ where $\kappa_{i}$ is the parameter of interest (control) and $u_{i}^{obs}$ refers to the observed data, which are obtained by computing the forward problem for a given $\kappa_{i}$ and adding noise to the forward solution. In other words, we have:

$$u^{obs}_{i} = \mathcal{F}(\kappa_{i}) + \varepsilon \quad \forall i \in [|1, n|]$$

where $\varepsilon$ is noise, and $\mathcal{F}$ is the forward operator that returns the solution of the correponding PDE for a given control $\kappa_{i}$.

For example, the following line will generate 500 training samples and 50 test samples for the heat time-independent forward problem (cf. section 5 paper). This will store this dataset named "heat_conductivity_500" into `./data/datasets`.

```generate_data
cd dataset_processing
python generate_data.py --forward heat --ntrain 500 --ntest 50 --dataset_name heat_conductivity_500
```

You can specify your own forward problem and custom noise by providing the corresponding callables in `./dataset_processing/generate_data.py`.

- Heat conductivity paper dataset

## Training

For training, we provide in `training/train_heat_conductivity.py` the code for training several models on the inverse heat conductivity example (cf. section 5 paper). This training script showcases how one can train PyTorch models with PDE components implemented in Firedrake. This example can easily be adapted to other forward problems by simply changing the PDE problem definition.

Several evaluation metric can be used for evaluation such as L2 or H1. For the paper experiments, we use an average L2-relative error across the test samples. The best performing model(s) with respect to the given evaluation metric is saved across the epochs. For example, the following command trains the model for 150 epochs on the heat conductivity dataset used in the paper using an average L2-relative error.

```training
cd training
python train_heat_conductivity.py --dataset heat_conductivity_paper --epochs 150 --model_dir cnn_heat_conductivity --evaluation_metric avg_rel
```

## Evaluation

For evaluation, we can leverage the full armoury of norms suited to PDE-based problems provided by Firedrake such as: L2, H1, Hdiv, or Hcurl. For the experiments, we employed an average relative L2-error norm (cf. section 5 paper). For inference, we need to specify the model directory as well as the model version corresponding to the saved model checkpoint of interest.

- model experiments.py

```evaluation
cd evaluation
python evaluate.py --model_dir cnn_heat_conductivity --model_version [...] --evaluation_metric avg_rel
```

## Citation
