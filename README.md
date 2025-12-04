# Tennis-Stroke-Classifier
CS5100 Project: Create a classifier for different tennis strokes using dinoV3, and analyze stroke form using pose estimation
# Prerequisites
To build the model, requires Dinov3 and MPII model weight files downloaded, file links: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, MPII files: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset
# Description
This project builds an image classifier using dinov3 as a backbone, training the model on a custom dataset of ~500 labeled images of US Open stroke images, which can be found in ./tennis
The tool also can evaluate an image of a tennis stroke based on the pose estimation (using MPII to locate body points) based on relevant angles for the stroke, comparing them to the average of the US Open stroke swing angles.
