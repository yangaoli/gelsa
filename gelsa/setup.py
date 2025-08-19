#License: BSD

#Copyright (c) 2008 Li Charles Xia
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#1. Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#3. The name of the author may not be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
#IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
#NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
#THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Local Simularity Analysis Package.

This python module provide tools for aligning and analyzing the time shift
dependent pairwise correlation between two sequence of evenly spaced 
observation data. Permutation test is used to estimate the P-value. 
The results can be summarized and easily queried for desired analysis.
"""

from setuptools import setup, find_packages
from distutils.command import build
import os, sys, subprocess

doclines=__doc__.splitlines()

git_on_cmd="echo 'def main():\n\t print' \"('$(cat VERSION.txt)' '@GIT: $(git log --pretty=format:'%h' | head -n 1)')\" > lsa/lsa_version.py" #lsa_version requires main() as an entry_point
git_on=subprocess.call(git_on_cmd, shell=True)

if os.path.exists('MANIFEST'): os.remove('MANIFEST')

class my_build(build.build):
    sub_commands = [('build_ext', build.build.has_ext_modules), 
        ('build_py', build.build.has_pure_modules), 
        ('build_clib', build.build.has_c_libraries), 
        ('build_scripts', build.build.has_scripts), ]

setup(name="lsa",
    version="1.0.2",
    description=doclines[0],
    long_description="\n".join(doclines[2:]),
    author="Li Charlie Xia",
    author_email="li.xia@stanford.edu",
    url="http://bitbucket.org/charade/elsa",
    license="BSD",
    platforms=["Linux"],
    packages=find_packages(exclude=['ez_setup', 'test', 'doc']),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8', 
    install_requires=["numpy >= 1.0", "scipy >= 0.6"], 
    provides=['lsa'],
    py_modules = ['lsa.lsalib'],
    cmdclass = {'build': my_build},
    data_files = [('',['LICENSE.txt','VERSION.txt'])],
    entry_points = { 'console_scripts': 
        [ 'lsa_compute = lsa.lsa_compute:main',
	  'm = lsa.m:main']
    },
)
