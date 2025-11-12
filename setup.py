from setuptools import setup, find_packages

setup(
    name='ec349-project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'transformers>=4.30',
        'scikit-learn>=1.3',
        'pandas>=1.5',
        'numpy>=1.23',
        'pytorch-lightning>=2.0',
        'pyarrow>=12.0',
        'python-dotenv>=1.0',
        'tqdm>=4.65',
        'pytest>=7.4'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for EC349',
    url='https://github.com/yourusername/ec349-project',
    python_requires='>=3.8',
)