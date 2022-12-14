ICVR_TEST
------------

Link EDA Notebook: https://nbviewer.org/github/Leotiv-Vibs/icvr_test/blob/master/eda.ipynb

Link Model sheet: https://docs.google.com/spreadsheets/d/11YPwpOxm0F1sxW4Zn66ICQTyNk6g98Dek8d67J-kEJY/edit?usp=sharing

Link Dataset sheet: https://docs.google.com/spreadsheets/d/11_SsFIici4v98D20GMl9OdwMgnBVfe2u_mgmFxn347Y/edit?usp=sharing

File description
------------
> eda.ipynb - a notebook for research and data comprehension 

> data_augmentation.py -  file to create an augmentation from the data. Organized as a class and has a user-friendly interface for using

> prepare_directory_train.py - file for creating directories and files to train the yolo model . Organized as a class and has a user-friendly interface for

> inference_new_data.py - a simple script for testing a model on a single image, followed by visualization and saving.

> settings.py - file with the number of classes and names of labels

> tools.py - auxiliary tools for work

> requirements.txt - file to create the desired virtual environment 

> img_result - directory with multiple results 


Inference model
------------
For quick and easy prediction from one simple image, you need to download the scales and specify the path to them in the variable "weights"

Download this weights: https://drive.google.com/file/d/1IJrP7Td6US1GKAUF7_ZeFWLJvFvBy7EK/view?usp=sharing


Change this variables in file "inference_new_data.py" and run script and be happy!
>path_image = 'PATH_TO_YOUR_IMAGE'
>
>path_save = 'PATH_TO_SAVE_IMAGE'
> 
>weights = 'PATH_TO_WEIGHTS' 

Check result
------------
Model - 1
--
Arch: yolov7

Dataset: https://drive.google.com/file/d/1HbiJmxtnMXPay4CJT2sG9iJc5OEkJCZi/view?usp=sharing

Metrics: https://drive.google.com/drive/folders/1GJANmU269uU814rkX1QGUqa9PZnlXLVg?usp=sharing

Weights: https://drive.google.com/file/d/1HbiJmxtnMXPay4CJT2sG9iJc5OEkJCZi/view?usp=sharing

Test_predict: https://drive.google.com/drive/folders/1oilDqXtTEzzY_dHgk5KhEFt_4sDHXqsA?usp=sharing

Batch_predict: https://drive.google.com/drive/folders/1FcFxjoJ6G4S0ICGRAmewLAp2luPWOkmW?usp=sharing


![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/results.png?raw=true)


Model - 2
--

Arch: yolov7

Dataset: https://drive.google.com/file/d/1ssqJyNiYOhoviqtd8cjiNegoiOh2o9ZZ/view?usp=sharing

Metrics: https://drive.google.com/drive/folders/1vGf5B8Uyv9CaIbfMKkQ2xkMR_ISou653?usp=sharing

Weights: https://drive.google.com/file/d/1IJrP7Td6US1GKAUF7_ZeFWLJvFvBy7EK/view?usp=sharing

Test_predict: https://drive.google.com/drive/folders/1iEZ8EUKr1QMfVkOnWZLiKOQbavNpHTNS?usp=sharing

Batch_predict: https://drive.google.com/drive/folders/188fOjx1hHnWRSpIrCfTOhWPhVnvvY9Km?usp=sharing


![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/results%20(1).png?raw=true)



Visualize
------------
![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/750.jpg?raw=true)

![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/999.jpg?raw=true)

![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/aug_8749.jpg?raw=true)

![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/aug_972.jpg?raw=true)

![alt text](https://github.com/Leotiv-Vibs/icvr_test/blob/master/img_result/pred.jpg?raw=true)

