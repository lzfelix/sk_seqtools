from setuptools import setup, find_packages

setup(
    name='sk_seqtools',
    version='0.0.1',
    author='Luiz Felix',
    author_email='lzcfelix@gmail.com',
    description='Extends sklearn LabelEncoder to handle sequences of labels',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.5',
        'sklearn>=0.0',
    ]
)
