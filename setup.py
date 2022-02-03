"""
This script is used to install the package and all its dependencies. Run

    python setup.py install

to install the package. Easier than the pip install thing.
"""

from setuptools import setup, find_packages

setup(
    name='UKAEA_Group_Project',
    version ='0.0.1',
    packages = find_packages(),
    url = "https://github.com/zcapjdb/UKAEAGroupProject"
)
