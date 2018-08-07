
# Markov Random Field For Image Segmentation and Denoising
This project has two parts. In part one, we use markov random field to denoise an image. In Part two, we use similar model for image segmentation. 


For a brief read me, click on [Brief Read me](README/README.md) <br>
For checking the code, click on [Codes](Codes/README.md)


$a=2$

Markov Random Field Models provide a simple and effective way to model the spatial dependencies in image pixels. <br>
So we useed them to model the connection between two neighbour pixels. <br>
In our problem we have to define an energy function on hidden states corresponding to true values of each pixels, then we minimize this function to obtain the best prediction. <br>
Our energy function is defined as below: <br>
 

($U(w)=\sum_{s} (\lg (\sigma_{\omega_{s}} \sqrt{2 \pi}) + \frac{(f_s - \mu_{\omega_{s}})^2}{2(\sigma_{\omega_{s}})^2}) + \sum_{s,r} \beta \delta (s,r)  $)

## Denoising


```python
a_complete_set_for_part_1(arr, max_iter=1e7, var=1e4, betha=1e4)
```


![png](output_4_0.png)



```python
plt.figure(figsize=(10, 12), dpi=80, facecolor='w', edgecolor='k')
known_index = np.zeros((len(arr), len(arr[0])))
for i in range (0, len(arr)):
    for j in range(0, len(arr[0])):
        if (i <= j ):
            known_index[i][j]=1
bta = 1e4
a_complete_set_for_part_1_some_pixels_known(arr,  known_index, max_iter=1e6, var=1e4, betha=bta)
```





![png](output_5_1.png)


## Image Segmentation
In this part, we used Markov Random Field for image segmentation.
<br>
We used different image color space:
- Gray Scale
- HSV
- RGB Format


### Gray Scale


```python
a_complete_set_for_part_2(arr,class_info, max_iter=1e7, betha=1e6)
```


![png](output_8_0.png)


### HSV color space

Now we want to use HSV color space for training our data.


```python
a_complete_set_for_part_2(arr_h,class_info, max_iter=1e6, betha=1e6)
```


![png](output_11_0.png)


### RGB color space
In this part, we used RGB color format in training since there is some information that can be captured by pixels colors.<br> 
We used RGB values in potential function.


```python
a_complete_set_for_part_2_3_color(max_iter=1e6, betha=1e6)
```


![png](output_13_0.png)



```python
a_complete_set_for_part_2_3_color(max_iter=1e6, betha=1e6,
                                 schedule=linear_multiplicative_cooling_schedule, temprature_function_constant=0.5)
```


![png](output_14_0.png)


## Conclusion
Grayscale image format didn't have sufficient information for CRF models in this task.<br>
The value H in HSV image format had better information for segmentation using CRF models. And the result was better. <br>
The RGB format also had good information for segmenting the image. Because these three segments have different colors. So if a CRF model considers colors of the image for classification, then the result is going to be better compared to Grayscale images.
