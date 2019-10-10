from setuptools import setup, find_packages


with open('requirements.txt') as reqs:
    install_requires = [line for line in reqs.read().split('\n') if (
        line and not line.startswith('--'))
    ]


setup(name='gcn',
      version='0.1',
      description='Framework for deep learning on graph structures',
      long_description_content_type='text/markdown',
      url='https://github.com/vojtechcima/gcn',
      author='Vojtech Cima',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires)
