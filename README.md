# Emotion Classifaction

## How to run

1. Clone this project

```
  git clone https://github.com/ske-13-emotion-classification/emotion-classification.git
  cd emotion-classification
  export ROOT_DIR=$PWD
```

2. Download OpenCV repository

```
  wget https://github.com/opencv/opencv/archive/master.zip
```

3. Unzip

```
  unzip master.zip
```

4. Create data in $ROOT_DIR & Move OpenCV data to \$ROOT_DIR/data

```
  mk -p $ROOT_DIR/data
  mv ./opencv-master/data $ROOT_DIR/data
```

5. Install dependencies

```
  pip install -r requirements.txt
```

6. Run Webcam emotion classification demo

```
  python main.py
```
