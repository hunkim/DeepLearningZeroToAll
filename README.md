== Lab code (WIP) == 

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