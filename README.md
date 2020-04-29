# MAV_assignment
Individual assignment to design an automatic gate detection method for Autonomous Drone Racing competitions. This assignment is for the course AE4317 - Autonomous Flight of Micro Air Vehicles. 

## Goal
The goal of this assignment is to create an automatic gate detection method for drones that are competing in Autonomous Drone Racing. A computer vision strategy / method must be found using Python or MATLAB that is most suited for the environment of Drone Racing. A trade-off analysis must be conducted in order to list the advantages and disadvantages of the variety of detection methods in the market. 

## Usage
This repository contains 2 python files and a map with figures that contains the test and template images. The python files are two detection methods that were tested and compared: these are the SURF and SIFT feature detection methods. Each of these python files can be run immediately after cloning or downloading this repository. 

The figure map only contains two figures that were tested. In case of more thorough testing, additional figure files should be uploaded in this map to test the detection algorithm.  

## The SURF.py
The python script of SURF.py contains two functions: one to process the SURF method on the template and test image and the second is to obtain the True Positive values (TP) and False Positive values (FP) to obtain the given ratios for a set of given data images. In the main, a SURF example is conducted which can be run immediately after cloning this repo. A commented section in the main is created to obtain the ROC curve. 

## The SIFT.py
The SIFT.py contains a simple variant of the SURF.py which only processes the SIFT method in order to compare the computational efforts of both methods, which is discussed in the essay. 


