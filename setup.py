import setuptools
with open("README.md", "r") as fh:
    setuptools.setup(
        name='gda-extraction',  
        version='0.1',
        author="Stefano Marchesin",
        author_email="stefano.marchesin@unipd.it",
        description="Exploiting Nanopublications to Generate Large-Scale Gene-Disease Association Datasets for Biomedical Relation Extraction",
        url="https://github.com/NanoGDA/gda-extraction",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
        ],
        setup_requires=['wheel']
     )
