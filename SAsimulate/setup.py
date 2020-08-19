from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='sasimulate',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Simulate stuck-at-faults neural network ",
    license="MIT",
    author="Nguyen Thai Hoang",
    author_email='hoangchuyenli@gmail.com',
    url='https://github.com/thnguyen996/sasimulate',
    packages=['sasimulate'],
    entry_points={
        'console_scripts': [
            'sasimulate=sasimulate.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='sasimulate',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
