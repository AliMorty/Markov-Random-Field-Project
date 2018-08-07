# Markov Random Field For Image Segmentation and Denoising
This project has two parts. In part one, we use markov random field to denoise an image. In Part two, we use similar model for image segmentation. 

## Part 1
In this part, we have an image. We add a gussian noise to it. Then we use markov model

### Original Image



```python
path = './test1.bmp'
arr = misc.imread(path, flatten=True)
labels = np.array(arr / 127, dtype=int)
print ("initial image")
imshow(arr, cmap='gray');
```

    initial image
    


![md](README/README.md)
