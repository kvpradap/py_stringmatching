============
Installation
============
This pages describes the requirements, dependencies and provides a step by step instruction
to install the py_stringmatching package.

Requirements
------------
    * Python 2.7 or Python 3.3+

Dependencies
------------
    * numpy>=1.7.0
    * six
    * python-Levenshtein >= 0.12.0


.. note::
The user need not install these dependency packages before installing the py_stringmatching package.
    The py_stringmatching installer will automatically install the required packages.


Step by Step Installation Instruction
-------------------------------------
Step 1: Download the py_stringmatching package from `here
<http://pradap-www.cs.wisc.edu/py_stringmatching/py_stringmatching-0.1.tar.gz>`_
into your home directory.

You can download into any directory within your home directory. For now we assume that you use a
linux operating system and will download into "HOME/", the top level.

Also, we assume that you have sufficient privileges to install a python package.

Step 2: Unzip the package by executing the following command::

    tar -xzvf py_stringmatching.tar.gz

py_stringmatching will be unpackaged into directory "HOME/py_stringmatching-0.1


Step 3: At the command prompt execute the following commands::

    cd HOME/py_stringmatching-0.1
    python setup.py install

This will install py_stringmatching package.

.. note::

    If the python package installation requires root permission then, you can install the package in
    your home directory like this::

        python setup.py install --user

    for more information look at the stackoverflow `link
    <http://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges>`_.

Supported Platforms
-------------------
It is tested primarily on OSX and Linux, but due to minimal dependencies it should work perfectly on Windows.
