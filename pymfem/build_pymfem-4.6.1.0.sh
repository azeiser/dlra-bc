# conda environment
eval "$(conda shell.bash hook)"
conda create -y -n pymfem-4.6.1.0 python=3.11 numpy scipy matplotlib cmake swig
conda activate pymfem-4.6.1.0

# mfem
git clone https://github.com/mfem/mfem.git
cd mfem
git checkout 4a45c70d1269d293266b77a3a025a9756d10ed8f  # from PyMFEM/setup.py -> repos_sha
git apply ../mfem-4.6.1.0.patch
mkdir mfem_build ; cd mfem_build
cmake .. -DBUILD_SHARED_LIBS=1 -DCMAKE_INSTALL_PREFIX="../mfem_install"
make -j 4
make install
cd ../..

# pymfem
git clone https://github.com/mfem/PyMFEM.git
cd PyMFEM
git checkout v_4.6.1.0
python setup.py install --mfem-source="../mfem/mfem_build" --mfem-prefix="../mfem/mfem_install"
