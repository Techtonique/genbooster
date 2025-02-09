from setuptools import setup, find_packages

setup(
    name="genbooster",
    version="0.6.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nnetsauce",
    ],
    python_requires=">=3.7",
    author="T. Moudiki",
    author_email="thierry.moudiki@gmail.com",
    description="A gradient boosting implementation with neural network features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Techtonique/genbooster",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)