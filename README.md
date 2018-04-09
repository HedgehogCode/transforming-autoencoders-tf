# TensorFlow implemetation of Transforming Auto-encoders

Implemenation of Transforming Auto-encoders (by Hinton et al.) using TensorFlow.

_Paper:
Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 2011._

## Getting Started

These instructions describe how I setup my environment for this project. There are different ways to get it running using `conda` or traditional `pip`.

### Prerequisites

- `python3`
- `pipenv`
- `jupyter`
- `cuda 9.0` (If you don't have a GPU, you can replace `tensorflow-gpu` by `tensorflow`)

### Installing

Clone the repository and install the python dependencies in a virtual environment using pipenv:
```
$ git clone git@github.com:HedgehogCode/transforming-autoencoders-tf.git
$ cd transforming-autoencoders-tf
$ pipenv install -d
$ pipenv shell
$ python -m ipykernel install --user --name transforming-autoencoders --display-name "Python (transforming-autoencoders)"
```

### Running

Start `jupyter notebook` and open a [notebook](nbs/).

## Similar Projects

- [yingzha/Transforming-Autoencoders](https://github.com/yingzha/Transforming-Autoencoders): Theano impemetation
- [nikhil-dce/Transforming-Autoencoder-TF](https://github.com/nikhil-dce/Transforming-Autoencoder-TF)
- [ndrplz/transforming-autoencoders](https://github.com/ndrplz/transforming-autoencoders)
- [ethanscho/TransformingAutoencoders](https://github.com/ethanscho/TransformingAutoencoders)
- ... (just search for Transfroming Autoencoders on GitHub)

## TODOs

* Add other datasets
  * [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) should be easy to use

## License

This project is licensed unter the 2-Clause BSD License - see the [LICENSE.txt](LICENSE.txt) file for details.
