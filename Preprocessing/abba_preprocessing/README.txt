ABBA PREPROCESSING STEPS
1) extract resized imgs from czi file
2) order sections
3) crop imgs (optional)
4) rotate/flip images (optional)
5) extract fullsize imgs + rename image files
6) configure animal data

ABBA steps
1) create a directory with a folder for each animal, the folder should be named just with the animal ID (e.g. 'animal1', 'animal2', ...)
2) create qupath project for each animal using those folders, add the fullsize ometiffs
3) align sections in abba
4) export...

Quant steps
1) stardist detections (and optimize)
2) extract region props
3) compile and filter counts (and optimize)
4) analyze

If you want to use the pre-trained model to predict image order you can download it here: ______ #TODO