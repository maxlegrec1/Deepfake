# Deepface attempts

This is my attempts at making relatively good deepface.
I base myself on the DeepFaceLab repo which was recently deleted.

The Deepfake technology is LIAE. Please read the DeepFaceLab paper, or look at the code for more information.


# Requirements

A good pytorch environement, I use very common librairies.
I Have used the following [repo](https://github.com/willyfh/farl-face-segmentation) to generate the masks for the images. 


## Files architecture
```
Project
│   README.md
│   all python files
└─dst/
│   │  put dst images here
│   └───masks/
│   │   put dst masks here
└─src/
│   │  put src images here
│   └───masks/
│   │   put src masks here
```

## Launch

```python train.py```
