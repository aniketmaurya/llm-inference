from setuptools import setup

with open("src/requirements.txt") as f:
    required = f.read().splitlines()

setup(
    install_requires=required,
)
