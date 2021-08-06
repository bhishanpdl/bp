from setuptools import setup

setup(
    name="bp",
    version="0.1",
    author="Bhishan Poudel",
    author_email="bhishanpdl@gmail.com",
    url = "https://github.com/bhishanpdl/bp",
    packages=["bp"],
    description="Personal library for data science.",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "pandas>=0.25"]
)