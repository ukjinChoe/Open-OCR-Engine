# OCR-Engine

__The Project is on going right now (2021-05-06)__

# Overview

Thanks to open source, it's not really difficult to build a customed OCR engine. In this repository, we'll do something below.

1. download and pre-process dataset for training model
2. try training text detection model and text recognization model respectively
3. custom your own OCR-Engine
4. serve your model on web

In [Google OCR Service Paper](https://das2018.cvl.tuwien.ac.at/media/filer_public/85/fd/85fd4698-040f-45f4-8fcc-56d66533b82d/das2018_short_papers.pdf#page=23), we can get a hint of building OCR Engine.

![](https://www.dropbox.com/s/zjkvt6cm3pv2f7x/google_ocr_structure.jpg?raw=1)  

__In this repository, we will leave `Direction ID`, `Script ID` and `Layout Analysis` parts empty.__ 

# Getting Started   

## 1. Data Generation   

First of all, we need to prepare training data. If you don't have good quality data, you can generate one. There are three steps to go.

A) Collect corpus.  

Locate your corpus in `./generate_data/texts/` directory. This corpus will be tokenized and renderd in the images of dataset. So, it would be best to gather corpus in target domain.
I recommend you to prepare more than 1MB of corpus as `.txt` file.  

__[Get Corpus](https://lionbridge.ai/datasets/the-best-25-datasets-for-natural-language-processing/)__          

B) Collect fonts.  

Locate your font files in `./generate_data/fonts/<lang>/` directory. The extension of font files should be `.otf` or `.ttf`. **Separate fonts by languages.** If your language is English the `<lang>` folder can be `en`.   

__[Get Fonts](https://www.dafont.com/)__

C) Generate dataset.  

```  
> cd generate_data
> python generate_data.py -i texts/my-corpus.txt -l en -c 880 -f 200 -rs -w 20 -t 1 -bl 2 -rbl -k 1 -rk -na 2 --output_dir out/
```

- `-i` : input corpus
- `-l` : language of fonts (language name in `generate_data/fonts` directory)
- __You can check all options in `generate_data.py`__

## 2. Train Text Detection Model

## 3. Train Text Recognition Model

## 4. Serve OCR Engine with API