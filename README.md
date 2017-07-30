Requirements
============
  * Python2.7
  * Tensorflow 1.1.0
  * Numpy
  * Scipy


Dataset
=======

VIPeR
-----
* Download the dataset from https://vision.soe.ucsc.edu/node/178
* Extract `VIPeR.v1.0.zip` file. You should get a VIPeR directory containing `cam_a` and `cam_b` folders.
* Run `viper/imgdump.py` script with `VIPeR` directory as the first CLI arg, and an output directory for storing the data batch files.
  Example:

      $ cd viper
      $ mkdir viper-data
      $ ./imgdump.py ~/Downloads/VIPeR.v1.0/VIPeR viper-data

  We'll be using `viper-data` for training/testing our models.


CUHK01
------
* Download the dataset from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
* Extract `CUHK01.zip` file. You should get a `campus` directory containing dataset image files.
* Run `cuhk01/imgdump.py` script with `campus` directory as the first CLI arg, and an output directory for storing the data batch files.
  Example:

      $ cd cuhk01
      $ mkdir cuhk01-data
      $ ./imgdump.py ~/Downloads/CUHK01/campus cuhk01-data

  We'll be using `cuhk01-data` for training/testing our models.



Training & Testing
==================
Both the models have `train.py` & `test.py` files for training & testing respectively. Both the files need two command-line args (in order):
- data dir: dataset dir path
- out dir: results dir path
    Used for storing model checkpoints and model statistics.

test.py supports an additional third CLI arg: `--run-once`; which evaluates the model accuracy on the test dataset once and exits. If `--run-once` is not specified, the script periodically (every 60s) evaluates the model accuracy on the test dataset. This behavior is helpful in observing the test accuracy trend against training steps.

Examples:
1. The following commands train the model and evaluate the trained model's performance on the test dataset:

          $ cd cuhk01
          $ ./train.py cuhk01-data out
          $ ./test.py cuhk01-data out --run-once


2. The following commands train and test the model simultaneously. Training in-progress model's performance on the test dataset is evaluated every 60s:

          $ cd viper
          $ ./train.py viper-data out &
          $ ./test.py viper-data out


Visualizations
==============
You can visualize all the parameters and results using TensorBoard during/after training. You can navigate to http://localhost:6006 after running the following command:

      $ python -m tensorflow.tensorboard --logdir=out

`out` is the directory path passed as the second CLI arg to `train.py` and `test.py` scripts.

Execution Times
===============
* VIPeR takes ~1 hour to converge on CPU based training. We haven't tried
  training on GPU.
* CUHK01 takes ~4.5 hours to converge on CPU based training. We haven't tried
  training on GPU.
