from pathlib import Path
from setuptools import setup, find_packages


# Read README's content
dir = Path(__file__).parent
long_description = (dir / "README.md").read_text()


setup(
    name="physics-driven-ml",
    version="0.1.0",
    description="Physics-driven machine learning using the Firedrake and PyTorch libraries",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Nacime Bouziani",
    author_email="n.bouziani18@imperial.ac.uk",
    packages=find_packages(),
    install_requires=["tqdm"],
)
