# create Python 3.7 venv

```
python3.7 -m venv ~/.py37env
```

## enter Python 3.7 venv

```
source ~/.py37env/bin/activate
```

## exit Python 3.7 venv

```
deactivate
```

# install tensorflow, keras

```
source ~/.py37env/bin/activate
pip install tensorflow
pip install pillow
pip install keras
```

# create rotate, blur images

```
source ~/.py37env/bin/activate
./bin/cv.sh
```
# prepare data

```
source ~/.py37env/bin/activate
python ./bin/read_images.py
```

# predict lcd image

![LCD image](https://github.com/eyasuyuki/tensor-led/blob/develop/images/example.jpg?raw=true)

```buildoutcfg
source ~/.py37env/bin/activate
python ./bin/canny.py
```

## result

```buildoutcfg
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
```