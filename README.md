# Physics-driven machine learning models

Examples of physics-driven machine learning models using PyTorch and Firedrake


## Setup

This work relies on the Firedrake finite element system, which needs to be installed.

### Installing Firedrake

Firedrake is installed via its installation script, which you can download by running:

```download_install_script
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
```

Then, you can install Firedrake and specify the required branches using:

```install_firedrake_pytorch_branches
python3 firedrake-install --package-branch firedrake pytorch_coupling --package-branch pyadjoint adjoint-1-forms
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
