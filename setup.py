from setuptools import setup


def get_requirements(file):
    with open(file) as f:
        required = f.read().splitlines()
    return required


required = get_requirements("requirements/requirements.txt")
dev_required = get_requirements("requirements/dev.txt")
extras = {"dev": dev_required}

setup(install_requires=required, extras_require=extras)
