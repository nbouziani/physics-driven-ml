
# import os
# import sys

from pathlib import Path
from setuptools import setup, find_packages
# from setuptools import setup, find_packages, Extension
# from Cython.Distutils import build_ext


# Read README's content
dir = Path(__file__).parent
long_description = (dir / "README.md").read_text()

# # List of Cython extensions
# cythonfiles = [("adjacency_dofs", ["petsc"])]


# def get_petsc_dir():
#     try:
#         petsc_dir = os.environ["PETSC_DIR"]
#         petsc_arch = os.environ.get("PETSC_ARCH", "")
#     except KeyError:
#         try:
#             petsc_dir = os.path.join(os.environ["VIRTUAL_ENV"], "src", "petsc")
#             petsc_arch = "default"
#         except KeyError:
#             sys.exit("""Error: Firedrake venv not active.""")

#     return (petsc_dir, os.path.join(petsc_dir, petsc_arch))


# petsc_dirs = get_petsc_dir()
# if os.environ.get("HDF5_DIR"):
#     petsc_dirs = petsc_dirs + (os.environ.get("HDF5_DIR"), )
# include_dirs = [np.get_include(), petsc4py.get_include()]
# include_dirs += ["%s/include" % d for d in petsc_dirs]
# dirs = (sys.prefix, *petsc_dirs)
# link_args = ["-L%s/lib" % d for d in dirs] + ["-Wl,-rpath,%s/lib" % d for d in dirs]

# extensions = [Extension("physics_driven_ml.utils.graph.{}".format(ext),
#                         sources=[os.path.join("physics_driven_ml", "utils", "graph", "{}.pyx".format(ext))],
#                         include_dirs=include_dirs,
#                         libraries=libs,
#                         extra_link_args=link_args) for (ext, libs) in cythonfiles]


setup(
    name="physics-driven-ml",
    version="0.1.1",
    description="Physics-driven machine learning using the Firedrake and PyTorch libraries",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Nacime Bouziani",
    author_email="n.bouziani18@imperial.ac.uk",
    packages=find_packages(),
    install_requires=["tqdm"],
    # cmdclass=dict(build_ext=build_ext),
    # ext_modules=extensions,
)
