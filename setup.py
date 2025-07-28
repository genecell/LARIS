from setuptools import setup, find_packages

setup(
    name='LARIS',
    version='0.1.0',
    description='LARIS for inference of Ligand And Receptor Interaction In Spatial transcriptomics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Min Dai, Tivadar Török, Dawei Sun',
    author_email='dai@broadinstitute.org, ttorok@broadinstitute.org, dsun@broadinstitute.org',
    url='https://github.com/genecell/zonetalk',
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False
)
