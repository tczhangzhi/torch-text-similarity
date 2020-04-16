from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from torch_text_similarity import __version__, __authors__
import sys

packages = find_packages()

def readme():
    with open('README.rst') as f:
        return f.read()

class PyTest(TestCommand):

    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(
    name='torch_text_similarity',
    version=__version__,
    license='MIT',
    description="Implementations of models and metrics for semantic text similarity. Includes fine-tuning and prediction of models",
    long_description=readme(),
    packages=packages,
    url='https://github.com/tczhangzhi/torch-text-similarity',
    author=__authors__,
    author_email='850734033@qq.com',
    keywords='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Topic :: Text Processing :: Linguistic',
        'Intended Audience :: Science/Research'
    ],
    install_requires=[
        'torch',
        'strsim',
        'fuzzywuzzy[speedup]',
        'pytorch-transformers==1.1.0',
        'scipy',
        'tqdm'
    ]
)