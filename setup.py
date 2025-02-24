import sys
from setuptools import setup, find_packages

sys.path.insert(0, "src")

import ir_eval

setup(
    name='ir_eval',  # Choose a name for your package
    version='0.0.0',  # Your package version
    package_dir={'': 'src'},  # Tell setuptools that packages are under 'src'
    packages=find_packages(where='src'),  # Find packages in the 'src' directory
    install_requires=[
        # List any dependencies your package needs
        'numpy',
    ],
)