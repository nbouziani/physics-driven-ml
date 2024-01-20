import os
import sys
import petsc4py
import numpy as np

from os import path
from setuptools import setup
from Cython.Build import cythonize

from Cython.Distutils import build_ext
import versioneer


try:
    from Cython.Distutils.extension import Extension
except ImportError:
    # No Cython Extension means no complex mode!
    from setuptools import Extension


def get_petsc_dir():
    try:
        petsc_dir = os.environ["PETSC_DIR"]
        petsc_arch = os.environ.get("PETSC_ARCH", "")
    except KeyError:
        try:
            petsc_dir = os.path.join(os.environ["VIRTUAL_ENV"], "src", "petsc")
            petsc_arch = "default"
        except KeyError:
            sys.exit("""Error: Firedrake venv not active.""")

    return (petsc_dir, path.join(petsc_dir, petsc_arch))


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext


# cython_compile_time_env = {'COMPLEX': complex_mode}
cythonfiles = [("adjacency_dofs", ["petsc"])]

petsc_dirs = get_petsc_dir()
if os.environ.get("HDF5_DIR"):
    petsc_dirs = petsc_dirs + (os.environ.get("HDF5_DIR"), )
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]
dirs = (sys.prefix, *petsc_dirs)
link_args = ["-L%s/lib" % d for d in dirs] + ["-Wl,-rpath,%s/lib" % d for d in dirs]

extensions = [Extension("{}".format(ext),
                        #"firedrake.cython.{}".format(ext),
                        #sources=[os.path.join("firedrake", "cython", "{}.pyx".format(ext))],
                        sources=["{}.pyx".format(ext)],#os.path.join("test", "{}.pyx".format(ext))],
                        include_dirs=include_dirs,
                        libraries=libs,
                        extra_link_args=link_args,
                        #cython_compile_time_env=cython_compile_time_env
                        )
             for (ext, libs) in cythonfiles]

setup(
    cmdclass=cmdclass,
    ext_modules=extensions,  # cythonize("dofs_adjacency_opt.pyx", "dofs_adjacency.pyx"),
)
