## Requirements
* colorama==0.4.6
* contourpy==1.3.1
* cycler==0.12.1
* fonttools==4.55.3
* kiwisolver==1.4.7
* llvmlite==0.43.0
* matplotlib==3.10.0
* numba==0.60.0
* numpy==2.0.2
* opencv-python==4.10.0.84
* packaging==24.2
* pillow==11.0.0
* pyparsing==3.2.0
* python-dateutil==2.9.0.post0
* scipy==1.14.1
* six==1.17.0
* tqdm==4.67.1

## To Resize Image 
With backward energy
```
python main.py -image image/{your_image_file_name} -resize -numOfDelete_H {# of height you want to shrink} -numOfDelete_W {# of width you want to shrink} 
```
With forward energy
```
python main.py -image image/{your_image_file_name} -resize -numOfDelete_H {# of height you want to shrink} -numOfDelete_W {# of width you want to shrink} -forward
```
For example: 
```
python main.py -image image/cat.jpg -resize -numOfDelete_H 200 -numOfDelete_W 200 
```
## To Enlarge Image
Run: 
```
python main.py -image image/beach.jpg -enlarge -numOfDelete_H 0 -numOfDelete_W {# of pixels you want add to width}
```
For example: 
```
python main.py -image image/beach.jpg -enlarge -numOfDelete_H 0 -numOfDelete_W 400
```
note: 
Currently, the program only support enlarge in width, so the value after "-numOfDelete_H" must be 0. 
## To Remove Object
You can use the image and the mask already in image folder. 
For example:
```
python main.py -image image/desert.jpg -mask image/desert_mask.png
```

All result will be written to the folder: out, with the same file name as input
