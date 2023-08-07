from setuptools import setup, find_packages

version = '0.0.19'
description = 'An open-source offline speech-to-text package for Bangla language.'

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

name = 'BanglaSpeech2Text'
author = 'sifat (shhossain)'
email = '<hossain@gmail.com>'

with open('requirements.txt') as f:
    required = f.read().splitlines()

keywords = ['python','speech to text','voice to text','bangla speech to text', 'bangla speech recognation', 'whisper model', 'bangla asr model', 'offline speech to text', 'offline bangla speech to text', 'offline bangla voice recognation', 'voice recognation']

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Text Processing :: Linguistic',
    'Topic :: Utilities',
    'Operating System :: OS Independent',
]

projects_links = {
    "Documentation": "https://github.com/shhossain/BanglaSpeech2Text",
    "Source": "https://github.com/shhossain/BanglaSpeech2Text",
    "Bug Tracker": "https://github.com/shhossain/BanglaSpeech2Text/issues",
}

# console scripts banglaspeech2text > cli.py > main
# bnsst

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=author,
    author_email=email,
    url="https://github.com/shhossain/BanglaSpeech2Text",
    project_urls=projects_links,
    packages=find_packages(),
    install_requires=required,
    keywords=keywords,
    classifiers=classifiers,
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'bnsst = BanglaSpeech2Text.cli:main',
        ],
    },
)
