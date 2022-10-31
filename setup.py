from setuptools import setup, find_packages

setup(
    name='deep_learning_playground',
    version='0.0.2',
    packages=find_packages(),
    url='https://github.com/Amerden/autopilot_pipeline',
    license='MIT',
    author='David Tonderski',
    author_email='dtonderski@gmail.com',
    description='A variety of smaller deep learning projects and '
                'implementations',
    install_requires=[
        'torch==1.13.0',
        'pytest>=7.2.0',
        'torchvision>=0.14.0'
    ]
)