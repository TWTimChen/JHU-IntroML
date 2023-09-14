from setuptools import setup, find_packages

setup(
    name='custom_ml_toolkit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'unittest'
    ],
)
