

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
