# Dynamical low-rank approximation of the Vlasov-Poisson equation with piecewise linear spatial boundary

[A. Uschmajew, A. Zeiser: Dynamical low-rank approximation of the Vlasov-Poisson equation with piecewise linear spatial boundary](https://arxiv.org/abs/2110.13481)


## Build patched MFEM + PyMFEM
Here we build the MFEM library with added CIP support with its Python wrapper PyMFEM:
- [MFEM](https://mfem.org/)
- [PyMFEM](https://github.com/mfem/PyMFEM)

The prerequisites are a working build system (c++, make, cmake). Furthermore we assume that [Anaconda](https://anaconda.org/) is used for Python. All steps are also performed in the file `pymfem/build_pymfem-4.6.1.0.sh`.

If you have a different setting you have to adapt the following steps.


### Anaconda

Create an Anaconda Python environment for the project
~~~
conda create -y -n pymfem-4.6.1.0 python=3.11 numpy scipy matplotlib cmake swig
conda activate pymfem-4.6.1.0
~~~


### Build patched MFEM
~~~
git clone https://github.com/mfem/mfem.git
cd mfem
git checkout 4a45c70d1269d293266b77a3a025a9756d10ed8f  
git apply ../mfem-4.6.1.0.patch
mkdir mfem_build ; cd mfem_build
cmake .. -DBUILD_SHARED_LIBS=1 -DCMAKE_INSTALL_PREFIX="../mfem_install"
make -j 4
make install
cd ../..
~~~

### Build PyMFEM
~~~
git clone https://github.com/mfem/PyMFEM.git
cd PyMFEM
git checkout v_4.6.1.0
python setup.py install --mfem-source="../mfem/mfem_build" --mfem-prefix="../mfem/mfem_install"
~~~


## Numerical experiments

All code is in the folder `src`.

### Activate environment

Before running any scripts first activate the virtual Python environment.
~~~
conda activate pymfem-4.6.1.0
~~~


### Triangle inflow

~~~
python triangle_inflow.py --id=1 --rauc_tau_base=1e-4 --level=0
python triangle_inflow.py --id=2 --rauc_tau_base=5e-5 --level=1
python triangle_inflow.py --id=3 --rauc_tau_base=2e-5 --level=2
~~~

### Landau Damping

#### Fixed Rank
~~~
python landau_damping.py --id=1 --dt=5e-3 --ode_step=3 --time_method=1 --rank=5
python landau_damping.py --id=2 --dt=5e-3 --ode_step=3 --time_method=1 --rank=10
python landau_damping.py --id=3 --dt=5e-3 --ode_step=3 --time_method=1 --rank=15
~~~

#### RAUC

~~~
python landau_damping.py --id=4 --dt=5e-3 --ode_step=3 --time_method=2 --rauc_max=20 --rauc_tau=1e-5
python landau_damping.py --id=5 --dt=5e-3 --ode_step=3 --time_method=2 --rauc_max=40 --rauc_tau=1e-6
python landau_damping.py --id=6 --rank=3 --level=1 --dt=1.25e-3 --time_method=2 --ode_step=3 --rauc_tau=1e-6 --rauc_max=40 --delta=1e-2
~~~


