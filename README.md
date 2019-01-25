# Emotion Classifaction

## How to run

1. Clone this project

```
  git clone https://github.com/ske-13-emotion-classification/emotion-classification.git
  export ROOT_DIR=$PWD
```

2. Download OpenCV repository

```
  wget https://github.com/opencv/opencv/archive/master.zip
```

3. Unzip

```
  unzip opencv-master.zip
```

3. Move OpenCV data to \$ROOT_DIR/data

```
  mv opencv-master/data $ROOT_DIR/data
```

4. Install dependencies

```
  pip install -r requirements.txt
```

5. Run Webcam emotion classification

```
  python main.py
```
