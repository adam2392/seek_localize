import os
import sys
from distutils.core import setup

from setuptools import find_packages

"""
To re-setup: 

    python setup.py sdist bdist_wheel

    pip install -r requirements.txt --process-dependency-links

To test on test pypi:
    
    twine upload --repository testpypi dist/*
    
    # test upload
    pip install -i https://test.pypi.org/simple/ --no-deps seek_localize

    twine upload dist/* 
"""

PACKAGE_NAME = "seek_localize"
with open(os.path.join('seek_localize', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'').strip('"')
            break
if version is None:
    raise RuntimeError('Could not determine version')
DESCRIPTION = "iEEG-BIDS anatomical and electrode coordinate interfacing software " \
              "for easily generating anatomical interpretations of iEEG data."
URL = "https://github.com/adam2302/seek_localize/"
MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.6
REQUIRED_PACKAGES = [
    "numpy>=1.19",
    "scipy>=1.6.0",
    "pandas>=1.0.3",
    "natsort",
    "nibabel>=3.2.0",
    "mne>=0.22.0",
    "mne-bids>=0.6",
    "pybv>=0.4.0",
    "nptyping"
]
CLASSIFICATION_OF_PACKAGE = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: Implementation",
    "Natural Language :: English",
]
AUTHORS = [
    "Adam Li",
    "Chester Huynh",
    "Christopher Coogan"
]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=version,
    description=DESCRIPTION,
    author=AUTHORS,
    long_description=open("README.rst").read(),
    # long_description_content_type='text/rst',
    url=URL,
    license="GNU General Public License (GPL)",
    packages=find_packages(exclude=["tests"]),
    project_urls={
        "Documentation": URL + "doc/",
        "Source": URL,
        "Tracker": URL + "issues",
    },
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    classifiers=CLASSIFICATION_OF_PACKAGE,
)
