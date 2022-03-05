# pix2pix
This is the implementation of "pix2pix".<br>
Original paper: P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. [link](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)

## Usage

### 1. Build
Please build the source file according to the procedure.
~~~
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ..
~~~

### 2. Dataset Setting

#### Recommendation
- CMP Facade Database<br>
This is a dataset of facade images assembled at the Center for Machine Perception, which includes 606 rectified images of facades from various sources, which have been manually annotated.<br>
Link: [official](http://cmp.felk.cvut.cz/~tylecr1/facade/)
Link: [Ready to use dataset](https://github.com/mrzhu-cool/pix2pix-pytorch)

#### Setting

Please create a link for the dataset.<br>
The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--trainI
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--trainO
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--validI
|    |--validO
|    |--testI
|    |--testO
|
|--Dataset2
|--Dataset3
~~~

You should substitute the path of training input data for "<training_input_path>", training output data for "<training_output_path>", test input data for "<test_input_path>", test output data for "<test_output_path>", respectively.<br>
The following is an example for "facade".
~~~
$ cd datasets
$ mkdir facade
$ cd facade
$ ln -s <training_input_path> ./trainI
$ ln -s <training_output_path> ./trainO
$ ln -s <test_input_path> ./testI
$ ln -s <test_output_path> ./testO
$ cd ../..
~~~

### 3. Training

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/train.sh
~~~
The following is an example of the training phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='facade'

./pix2pix \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --loss "vanilla" \
    --batch_size 1 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/train.sh
~~~

### 4. Test

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/test.sh
~~~
The following is an example of the test phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='facade'

./pix2pix \
    --test true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --input_nc 3 \
    --output_nc 3
~~~
There are no particular restrictions on both input and output images.<br>
However, the two file names must correspond without the extension.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~


## Acknowledgments
This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
