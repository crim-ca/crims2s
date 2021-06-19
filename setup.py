from setuptools import setup, find_packages

setup(
    name="crims2s",
    author="CRIM",
    author_email="david.landry@crim.ca",
    install_requires=["hydra-core"],
    entry_points={},
    packages=find_packages(exclude="test"),
    include_package_data=True,
)
