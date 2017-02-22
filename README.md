# Lab code (WIP)
This is work in progress. Please do not use them, since they may have many bugs and trial code. We will let you know when it's done.

## Naming rule:

* klab-XX-X-[name].py: Keras labs
* lab-XX-X-[name].py: regular tensorflow labs


## Run test and autopep8

```bash
python -m unittest discover -s tests;

# http://stackoverflow.com/questions/14328406/
pip install autopep8 # if you haven't install
autopep8 . --recursive --in-place --pep8-passes 2000 --verbose
```
