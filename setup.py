from setuptools import setup, find_packages

setup(
    name='modi-flows',
    version='0.1.5',
    description='MODI: Multicommodity Optimal Transport-based Dynamics for Image Classification',
    author=['Alessandro Lonardi','Diego Baptista'],
    author_email='diego.theuerkauf@tuebingen.mpg.de',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'networkx',
        'scipy',
        'scikit-image',
        'jupyter',
        'nbimporter',
        'pandas',
        'tqdm',
        'Pillow',
        'matplotlib',
        #'scikit-umfpack'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
