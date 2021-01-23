# Python Labeled Multi-Bernouli Filter Implementation
This is an implementation of the Labeled Multi-Bernoulli filter,
implemented for educational purposes and for the purpose of the article

Olofsson, J., Veibäck, C., & Hendeby, G. (2017). Sea ice tracking with a spatially indexed labeled multi-Bernoulli filter. In 20th International Conference on Information Fusion (FUSION). Xi’an, China.

# Usage
=====
Install packages: `numpy`, `scipy`, `matplotlib`, `lapjv`, `shapely`

Install `lapjv` by `cd ./deps/lapjv-0.03/python-lapjv` then `python setup.py build` and `python setup.py install`.

Download MOT20 dataset from https://motchallenge.net/data/MOT20/.

Copy MOT20/test/MOT20-04 to the root folder of this source code.

Create `output` folder inside MOT20-04 to store output track images.

Run: `python demo_mot20.py`

# License
Refer to the original [lmb](https://github.com/jonatanolofsson/lmb) repository by [danstowell](https://github.com/jonatanolofsson).
