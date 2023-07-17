from setuptools import setup, find_packages

setup(
    name='modi',
    version='0.1.0',
    description='MODI: Multicommodity Optimal Transport-based Dynamics for Image Classification',
    author='Alessandro Lonardi',
    author_email='alessandro.lonardi.vr@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.20.0',
        'networkx==2.5.1',
        'scipy==1.7.0',
        'scikit-image==0.18.3',
        'jupyter',
        'nbimporter',
        'pandas',
        'tqdm',
        'Pillow',
        'matplotlib',
        'scikit-umfpack'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
