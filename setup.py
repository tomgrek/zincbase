from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name='Zincbase',
    version='0.3.1',
    packages=setuptools.find_packages(),
    description="A state of the art knowledge base and batteries-included NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    include_package_data=True,
    url='https://github.com/tomgrek/zincbase',
    author='Tom Grek',
    author_email='tom.grek@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
