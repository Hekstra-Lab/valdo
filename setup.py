from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("valdo/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

setup(name="rs-valdo",
    version=__version__,
    author="Phyllis Zhang, Minhuan Li",
    license="MIT",
    description="Vae assisited ligand discovery for crystallographic fragment screening", 
    url="https://github.com/Hekstra-Lab/drug-screening",
    author_email='phylliszhang@college.harvard.edu, minhuanli@g.harvard.edu',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.13.0",
        "reciprocalspaceship>=0.9.18",
    ],
)
