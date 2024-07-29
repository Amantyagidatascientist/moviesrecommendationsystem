from setuptools import  find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return this list of requirements

    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[ req.replace('\n'," ")for req in requirements]
    return requirements


setup(
name='mlproject',
author='Aman Tyagi',
version='0.0.1',
author_email='tyagiaman3558@gmail.com',
install_requires=get_requirements('requirements.txt'),
packages=find_packages()
)