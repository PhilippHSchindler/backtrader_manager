from setuptools import setup, find_packages

setup(
    name='backtrader_manager',
    version='0.1.0',
    description='Framework for organizing and managing backtests with Backtrader, including walk-forward optimization.',
    author='Philipp Schindler',
    author_email='Philipp.H.Schindler@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'backtrader',   # Add other dependencies here
        'pandas', 
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # or whatever you prefer
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.1',
)
