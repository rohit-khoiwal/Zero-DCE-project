# Zero-DCE-project

## Introduction
Low-light picture enhancement is defined by Zero-Reference Deep Curve Estimation (Zero-DCE) as the challenge of predicting an image-specific tonal curve using a deep neural network.
In this demonstration, we train the lightweight deep network DCE-Net to calculate high-order tonal curves at the pixel level for the purpose of adjusting the dynamic range of a given image. 

## Architecture
|   |   |
|---|---|
| ![framework](https://user-images.githubusercontent.com/87682045/192520372-0207f6b0-c7ac-4448-bc8a-5345972b497e.png)|![layers](https://user-images.githubusercontent.com/87682045/192520494-b195da62-3f03-4716-8e3d-e100dcf36257.png)  
|||

## Getting Started
It strictly need linux environment to run JAX and FLAX Framework.
- Clone the repo.
- Add dark/low-light image to test folder.
- Open Terminal
Then execute the following command one at a time.
```python
  cd Zero-DCE-project
  pip install -r requirements.txt
  python app.py
```


## Results
| Before | After |
|  ----  | ----- |
|![1](https://user-images.githubusercontent.com/87682045/192528300-7f7c7f7c-7439-4d71-a68f-ef45e878c60d.png)|![enhance_1](https://user-images.githubusercontent.com/87682045/192528424-2aa6a52f-d86e-4649-b1a6-6cdedb1e4b43.png)|
|![2](https://user-images.githubusercontent.com/87682045/192528374-6dd72a89-9d10-4dd7-8082-afa7703faad3.png)|![enhance_2](https://user-images.githubusercontent.com/87682045/192528483-feee73e0-83d8-4dc4-a9bf-86d2d5afc2d4.png)|

## References
  - [JAX](https://jax.readthedocs.io)
  - [Flax](https://flax.readthedocs.io)


## Citation

```
@article{2001.06826,
    Author = {Chunle Guo and Chongyi Li and Jichang Guo and Chen Change Loy and Junhui Hou and Sam Kwong and Runmin Cong},
    Title = {Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
    Year = {2020},
    Eprint = {arXiv:2001.06826},
    Howpublished = {CVPR 2020},
}
