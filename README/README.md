
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
    


![png](output_3_1.png)




Markov Random Field Models provide a simple and effective way to model the spatial dependencies in image pixels. <br>
So we useed them to model the connection between two neighbour pixels. <br>
In our problem we have to define an energy function on hidden states corresponding to true values of each pixels, then we minimize this function to obtain the best prediction. <br>
Our energy function is defined as below: <br>
 

![formula](formula.bmp)

## Naive Bayes for initial values
We used naive bayes algorithm to define an initial state for the true pixels values.<br>
## Markov Random Field using Simulated Annealing
Then we used simulated annealing optimization method. We tested different hyper pararmeters to compare the performance of the MRF models with  naive bayes model. <br>


```python
a_complete_set_for_part_1(arr, max_iter=1e7, var=1e4, betha=1e4)
```


![png](output_7_0.png)


### Model with four neighbors


```python
plt.figure(figsize=(10, 12), dpi=80, facecolor='w', edgecolor='k')
a_complete_set_for_part_1(arr, max_iter=1e6, var=1e4, betha=1e4)
```





![png](output_9_1.png)


### Model with eight neighbors


```python
eight_neighbors_indices =  [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
a_complete_set_for_part_1(arr, max_iter=1e6, var=1e4, betha=1e4, neighbor_indices=eight_neighbors_indices)
```


    <matplotlib.figure.Figure at 0x835cd53d30>



![png](output_11_1.png)


### Comparision
When we used eight neighbours, the smoothness of result increased. The weakness of eight neighbors parameter is that the model can't find a good boundary for the circle in the center. Because circles border are different from rectangles and so the number of same color in the border of circle is less than rectangles border. And the model connot predict well in this situation. If we use lower betha, then the result for the circle is going to be better.

### Accuracy - Different $\beta$ 


```python
xs = [-4,-3,-2,-1,0,1,2,4]
accs = [0.623, 0.627, .633, .729, .8587, .8581, .8552, .8551]
plt.plot(xs, accs)
plt.show()
```


![png](output_14_0.png)


According to the above graph, the best $\beta$ is 1e0=1.

## Result when some part are Observed
In this part, we have some part of the image and we want to predict the remaining part. 


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


    <matplotlib.figure.Figure at 0xa6d9bbad68>



![png](output_17_1.png)


## Different Schedule in Simulated Annealing
In this part, we compared different schedules in simulated annealing optimizer.<br>
We tested different scheduling Strategy. And the result is as follow.

#### Exponential Schedule


```python
a_complete_set_for_part_1(arr, max_iter=1e6, var=1e4, betha=bta,
                          schedule=exponential_schedule, temprature_function_constant=0.8)
```


![png](output_20_0.png)


#### Logarithmical Multiplicative Cooling Schedule


```python
a_complete_set_for_part_1(arr, max_iter=1e6, var=1e4, betha=bta,
                          schedule=logarithmical_multiplicative_cooling_schedule, temprature_function_constant=1.0)
```


![png](output_22_0.png)


#### Linear Multiplicative Cooling Schedule


```python
a_complete_set_for_part_1(arr, max_iter=1e6, var=1e4, betha=bta,
                          schedule=linear_multiplicative_cooling_schedule, temprature_function_constant=1.0)
```


![png](output_24_0.png)


## Image Segmentation
In this part, we used Markov Random Field for image segmentation.
<br>
We used different image color space:
- Gray Scale
- HSV
- RGB Format

### Orginal Image


```python
path = "test2-mini.jpg"
arr = misc.imread(path, flatten=True)
print ("initial image")
imshow(arr, cmap='gray');
```

    initial image
    


![png](output_27_1.png)



```python
tmp = plt.gcf().clear()
```


    <Figure size 432x288 with 0 Axes>



### Gray Scale


```python
a_complete_set_for_part_2(arr,class_info, max_iter=1e7, betha=1e6)
```


![png](output_30_0.png)


### HSV color space

Now we want to use HSV color space for training our data.


```python
import matplotlib
path = "test2-mini.jpg"
img = Image.open(path)
hsv_arr = matplotlib.colors.rgb_to_hsv(img)
hsv_arr = np.asarray(hsv_arr)
h_arr = hsv_arr[:,:,0]
plt.imshow(h_arr, cmap="gray")
plt.show()
```


![png](output_33_0.png)



```python
a_complete_set_for_part_2(arr_h,class_info, max_iter=1e6, betha=1e6)
```


![png](output_34_0.png)


### RGB color space
In this part, we used RGB color format in training since there is some information that can be captured by pixels colors.<br> 
We used RGB values in potential function.


```python
a_complete_set_for_part_2_3_color(max_iter=1e6, betha=1e6)
```


![png](output_36_0.png)



```python
a_complete_set_for_part_2_3_color(max_iter=1e6, betha=1e6,
                                 schedule=linear_multiplicative_cooling_schedule, temprature_function_constant=0.5)
```


![png](output_37_0.png)


## Conclusion
Grayscale image format didn't have sufficient information for CRF models in this task.<br>
The value H in HSV image format had better information for segmentation using CRF models. And the result was better. <br>
The RGB format also had good information for segmenting the image. Because these three segments have different colors. So if a CRF model considers colors of the image for classification, then the result is going to be better compared to Grayscale images.
