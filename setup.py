from pathlib import Path

from setuptools import setup

readme_path = Path(__file__).absolute().parent.joinpath('README.md')

with readme_path.open('r', encoding='utf-8') as fh:
    long_description = fh.read()

requirements_path = Path(__file__).absolute().parent.joinpath('requirements.txt')

with requirements_path.open('r') as f:
    requirements = f.readlines()
    requirements = [req for req in requirements if "--hash" not in req]
    requirements = [req.split("\\")[0].split(":")[0].strip() for req in requirements]

setup(
    name='cinnamon_generic',
    version='0.1',
    author='Federico Ruggeri',
    author_email='federico.ruggeri6@unibo.it',
    description='[Tensorflow Package] A simple high-level framework for research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/federicoruggeri/cinnamon_tf',
    project_urls={
        'Bug Tracker': "https://github.com/federicoruggeri/cinnamon_tf/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    license='MIT',
    packages=['cinnamon_tf',
              'cinnamon_tf.components',
              'cinnamon_tf.configurations',
              ],
    install_requires=requirements,
    python_requires=">=3.6"
)
