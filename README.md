# NeRF-101-Doc


<!-- In lieu of your final project, please take the NeRF 101 markdown and create a new markdown with a lot more details, including initial steps (read and summarize the NeRF paper, create CCV account, etc.) all the way to getting it to work on BRICS data. I expect this will be quite detailed: the equivalent of 10 pages of so of a regular report since you are all collaborating.
So we are all on the same page, could you send me a structure for this markdown file (just subsections) before next week's group meeting? -->

<!-- ## NeRF Paper
### Summary
   -->
    


## NeRF 101
Here is the link to the original NeRF paper: [NeRF](https://arxiv.org/abs/2003.08934)
### What's NeRF?
NeRF which stands for Neural Radiance Field is a fully connected neural network that can render novel and realistic views of complex 3D scenes from an input of a partial collection of 2D images of the same. The algorithm works by taking the partial set of 2D images and interpolating between them to render one complete 3D scene. The network is trained to map from a 5D coordinate system of viewing direction and spatial location to a 4D output of volume and opacity. NeRF is indeed a computationally intensive algorithm and hence there is a need to parallelize certain aspects of it using GPUs. In fact, one of the leaders of NeRF technology is Nvidia.

For a more detailed introductory insight into the workings and various components of NeRF please refer to this 
[website](https://datagen.tech/guides/synthetic-data/neural-radiance-field-nerf)






## About NeRF-Pytorch
The repository linked below is a faithful PyTorch implementation of the above described NeRF paper that reproduces the results of the author's Tensorflow implementation while running 1.3 times faster. It has also been tested to match it numerically.

This code is taken from the [nerf-pytorch repo](https://github.com/yenchenlin/nerf-pytorch). In order to run the code, you can reference the documentation in that repo and the breif desrciption below. 

NeRF-Pytorch can take in several different types of scenes as input (refer to the different `load_{data type}.py` files in the repo). We only dealt with loading synthetic data. The synthetic data we used was created by Chandradeep in Blender.

## Batch Script Command
`python run_nerf.py --config configs/<config-file>.txt`

## Config File
Inside <config_file>.txt, we define the file path, data type, training setting, etc.

### Location
```
├── configs
│    └── <config_file>.txt
```
### Content
This config file is based on the synthetic blender data we used.
```
expname = blender_<dataset name>
basedir = ./logs
datadir = ./data/nerf-synthetic/<dataset_name>

no_batching = True
use_viewdirs = True
lrate_decay = 500

N_samples = 64
N_importance = 128
N_rand = 1024

precrop_iters = 500
precrop_frac = 0.5
```

## Dataset
NeRF-Pytorch requires as input RGB images from each camera view and their associated camera data (intrinsics and extrinsics). 
### Folder Setup
```
├── test
│    └── <image from dataset>
├── train
│    └── <image 0>
│    ...
│    └── <image n>
├── transformations_train.json
├── val
│    └── <image from dataset>
├── transformations_test.json
├── transformations_val.json
```
### transformations_train.json
(transformation matrices for images in the train folder)
```
{
  'camera_angle_x': <camera_angle>
  'frames': [ { 'file_path': <image 0 file path>, 
                'transformation_matrix': <4x4 matrix> }, 
              ...
              { 'file_path': <image n file path>, 
                'transformation_matrix': <4x4 matrix> } ]
}
```
### transformations_test.json / transformation_val.json
(transformation matrices for images in the test and val folders)
```
{
  'camera_angle_x': <camera_angle>,
  'frames': [ { 'file_path': <image file path>, 
                'transformation_matrix': <4x4 matrix> } ]
}
```


## Intended versions of all programs

- As found in requirements.txt. Description of compatible cuda toolkit
- Resources and guidance on matching versions for intial run 
- Include additional pytorch information for those who haven't used it before

## Complete overview of BRICS/NeRF pipeline

- High level description of what each component does, inputs and outputs.
- Each section below will contain more detailed descriptions of these parts.
- Awesome diagram maybe?!?!?! Multiple diagrams o_o

## Run NeRF with example data

- Step by step guide with tips on how to generate fern output
- How to adjust parameters to get good output in a reasonable amount of time

## Run COLMAP with example data

- Links and description of colmap
- Detailed account on its role in the pipeline, expected outputs
- Colmap usage with NeRF
- Theoretical description of individual components, such as poses_bounds, sparse, images_n, etc.

## Collecting BRICS data

- Guide on usage, system, dos and don'ts, tips on how to get the best output.
- Theoretical goals of BRICS! Fun features!

## Run NeRF with BRICS Data
Details on how to prepare the BRICS data for nerf and provide examples.

- How to formulate BRICS images for colmap input
- Generate NeRF model of collected data using above info

## Run NeRF on Brown CCV 
Brown CCV is a campus-wide high performance computing cluster. All Brown member can apply for an exploratory account for free. Check out [here](https://ccv.brown.edu/rates) for your available resources. 

### Create an Account on CCV
1. Submit a request [here](https://brown.co1.qualtrics.com/jfe/form/SV_0GtBE8kWJpmeG4B). CCV staff will notify you via email after your account has been created (may take few days).
2. One recommended way to connect to CCV is by Remote IDE. You can find the instruction [here](https://docs.ccv.brown.edu/oscar/connecting-to-oscar/remote-ide) as well as some other connection methods.

### Submitting job on CCV
Brown CCV uses Slurm to manage workload. A shell script is required for submitting jobs. Below is an example.

```
#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
#SBATCH --mem=4G

# Specify a job name:
#SBATCH -J nerf-101

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o MySerialJob-%j.out
#SBATCH -e MySerialJob-%j.out

# number of gpus to use
#SBATCH -p gpu --gres=gpu:2

# number of nodes to use
#SBATCH -N 1

# number of cpu cores to use
#SBATCH -n 4

# your command
echo "nerf-101"
python run_nerf.py --config configs/<config-file>.txt

```

## Troubleshooting/Common issues in each process
Description of various user-error pitfalls encountered during the process by everyone

## Cylindrical/Corkscrew Path
This is the code can be found in the `load_blender.py` file. 

## Color Depth Mapping
The color depth mapping code (the `color_map.py` file) was taken from the Nerfies code found [here](https://github.com/google/nerfies). You can change the color depth mapping back to the regular depth mapping in lines 173-180 in the `run_nerf.py` file. 

## Problems Encountered
- For synthetic data, the near and far planes are hard-coded in lines 622 and 623 in the `run_nerf.py` file. The original near and far planes are set to 2 and 6 respectively. For Chandradeep’s data, the near and far planes are 0.1 and 20 respectively.
- The original code output images and a video where the camera moves in a circular path around the object. For the synthetic data, there were images (and parts of the video) that were completely white. We believe that this occurs because camera goes outside the bounds of the box, so decreasing the radius the camera movement resolves this issues.
- The code uses the Blender coordinate system rather than the OpenCV coordinate system. 





