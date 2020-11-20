# floodDetection using Semantic Segmentation with auxiliary data (Height Above Nearest Drainage)

In this Git, I've listed the py scripts used to build the dataset with OSM labels, however wouldn't recommend using that dataset instead I recommend using the dataset where I've manually selected flooded images and hand labeled. The reason being the model trained on non flooded OSM labels didn't generalize well to the disaster images for many reasons.

# Hand selected dataset: https://drive.google.com/file/d/1MbbEq3ORQYm8JmkG1U86UCMKGkfWZX6K/view?usp=sharing

The two encoder U-Net hasn't been run yet due to the size of the neural network; likely will have to move away from the ResNet architecture to something more lightweight or the second encoder will be a lightweight path.
