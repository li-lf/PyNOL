from setuptools import setup, find_packages

setup(
    name='pynol',
    version='0.1',
    description='Python Package for Non-Stationary Online Learning',
    packages=find_packages(),
    url='https://github.com/li-lf/PyNOL',
    license='MIT License',
    author='Long-Fei Li, Peng Zhao, Yan-Feng Xie, Lijun Zhang, Zhi-Hua Zhou',
    author_email='lilf@lamda.nju.edu.cn',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        'Topic :: Software Development :: Libraries'
    ],
    install_requires=[
        'numpy',
        'autograd',
        'cvxpy',
        'matplotlib',
    ])
