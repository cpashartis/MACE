from setuptools import setup, find_packages

setup(
    name='mace',
    version='0.1.0',
    author='Christopher Pashartis',
    description='A mesh optimization package using simulated annealing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cpashartis/MACE.git',
    packages=find_packages(exclude=('tests*',)),
    install_requires=[
        'numpy>=1.19.2',
        'scipy>=1.5.2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.7',
    keywords='mesh optimization, simulated annealing, spatial queries',
)
