from setuptools import setup

setup(
    name='drake-pytorch',
    version='0.1',
    packages=['drake_pytorch',],
    install_requires=[
        'drake',
        'torch'
    ]
)
