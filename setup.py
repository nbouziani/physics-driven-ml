from setuptools import setup, find_packages

setup(
    name="physics-driven-ml",
    version="0.1.0",
    description="Physics-driven machine learning using the Firedrake and PyTorch libraries",
    author="Nacime Bouziani",
    author_email="n.bouziani18@imperial.ac.uk",
    packages=find_packages(),
    install_requires=["tqdm"],
)
