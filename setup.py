import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="consensus-stefano",
    version="0.0.1",
    author="Savazzi Stefano",
    author_email="stefano.savazzi@ieiit.cnr.it  ",
    description="Package for consensus based federated learning tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/labRadioVision/federated",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)