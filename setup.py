try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'ClusterSVDD--Latent variable support vector data description',
    'url': 'https://github.com/nicococo/ClusterSVDD',
    'author': 'Nico Goernitz',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '0.1',
    'install_requires': ['numba', 'cvxopt','scikit-learn','numpy', 'scipy'],
    'packages': ['ClusterSVDD'],
    'package_dir' : {'clusterSVDD': 'ClusterSVDD'},
    #'package_data': {'clusterSVDD': ['*.txt']},
    #'scripts': ['bin/ClusterSVDD.sh'],
    'name': 'ClusterSVDD',
    'classifiers':['Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7']
}

setup(**config)