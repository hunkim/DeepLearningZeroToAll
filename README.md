# Lab code (WIP), but call for comments
This is code for labs covered in TensorFlow basic tutorials (in Korean) at https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm. 
(We also have a plan to record videos in English.)

This is work in progress, and may have bugs. 
However, we call for your comments and pull requests. Check out our style guide line:

* More TF (1.0) style: use more recent and decent TF APIs.
* More Pythonic: fully leverage the powe of python
* Readability (over efficiency): Since it's for instruction purposes, we prefer *readability* over others.  
* Understandability (over everything): Understanding TF key concepts is the main goal of this code.
* KISS: Keep It Simple Stupid! https://www.techopedia.com/definition/20262/keep-it-simple-stupid-principle-kiss-principle
 
## File naming rule:

* klab-XX-X-[name].py: Keras labs code
* lab-XX-X-[name].py: TensorFlow lab code

## Install requirements
```bash
pip install -r requirements.txt
```

## Run test and autopep8
TODO: Need to add more test cases

```bash
python -m unittest discover -s tests;

# http://stackoverflow.com/questions/14328406/
pip install autopep8 # if you haven't install
autopep8 . --recursive --in-place --pep8-passes 2000 --verbose
```
## Automatically create requirements.txt

```bash
pip install pipreqs

pipreqs /path/to/project
```
http://stackoverflow.com/questions/31684375

# Contributions/Comments
We always welcome your comments and pull requests.
