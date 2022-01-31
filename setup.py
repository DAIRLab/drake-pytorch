from setuptools import setup
install_reqs = ['torch']
try:
    import pydrake
    print('USING FOUND DRAKE VERSION')
except ModuleNotFoundError as e:
    install_reqs += ['drake']

setup(
    name='drake-pytorch',
    version='0.1',
    packages=['drake_pytorch',],
    install_requires=install_reqs
)
