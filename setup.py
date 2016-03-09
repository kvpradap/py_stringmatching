from setuptools import setup

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.



import sys



setup(
        name='py_stringmatching',
        version='0.1',
        description='Python library for string matching.',
        long_description="""
    String matching is an important problem in many settings such as data integration, natural language processing,etc.
    This package aims to implement most commonly used string matching measures.
    """,
        url='http://github.com/kvpradap/py_stringmatching',
        author='Pradap Konda',
        author_email='pradap@cs.wisc.edu',
        license=['MIT'],
        packages=['py_stringmatching'],
        install_requires=[
            'numpy >= 1.7.0',
            'six',
            'python-Levenshtein >= 0.12.0'
        ],
        include_package_data=True,
        zip_safe=False
)