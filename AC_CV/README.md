# Assetto Corsa Computer Vision - Writeup

<img src="https://github.com/kek01101/AC_computer_vision/blob/main/AC_CV/Picture1.png" height=80/>

## Abstract

A true self-driving vehicle for Assetto Corsa has never fully been implemented before. Former projects have been able to provide accurate information and input prediction, but the barrier to actually controlling the vehicle has never before been crossed. However, using exceptionally imprecise keyboard controls, it was found that a neural network was able to drive a vehicle in Assetto Corsa while solely being trained off of normal keyboard inputs. This demonstrates that in the future, a computer-vision based fully autonomous vehicle within Assetto Corsa could be designed. A road map for doing so can be seen in the discussion.

## Project Presentation
You can find the project presentation video [here](https://youtu.be/1ZoNVyIvUuI).

## Introduction

Assetto Corsa is a reasonably popular racing simulator that I like playing from time to time. I wanted to create an autonomous racing car for Assetto as a proof of concept. The racing car was supposed to function at Level 4, and was going to be designed to be used on race tracks only. The challenges the car were to be tested on were lane detection and obstacle (other car) avoidance, and if time allowed, actual racing techniques such as determining racing lines, optimal engine braking, and maybe even overtaking.

However, due to time constraints and issues with Jetracer, the project instead morphed into a Neural Network which was trained upon racing inputs, and created racing outputs.

### Project Objectives
  
<img src="https://github.com/kek01101/AC_computer_vision/blob/main/AC_CV/Picture3.png" height=80/>

### Level of Autonomy

The AC CV vehicle was originally intended to be an autonomous vehicle functioning at level 4 or 5 autonomy, meaning it can drive without the aid of intervention of a human. In practice, the results became a level 4 autonomous vehicle. However, if you include the fact that the car must manually be up-shifted for the autonomous part to start, the car may in fact only be at level 2.

## System overview

### Actual system process

<img src="https://github.com/kek01101/AC_computer_vision/blob/main/AC_CV/Picture4.png" height=80/>

<img src="https://github.com/kek01101/AC_computer_vision/blob/main/AC_CV/Picture5.png" height=80/>

Pedal and wheel inputs are split into two separate files in order to avoid confusing the neural network. Hence, there are two separate datasets upon which two separate models are trained.

### Neural Network Architecture

<img src="https://github.com/kek01101/AC_computer_vision/blob/main/AC_CV/Picture6.png" height=80/>

Note that the embedding and hidden layer are actual 8x larger in practice, the real sizes were too big for the visualizer. 

NN layout:
* Embedding dim: 256 nodes
* RNN layer: 1024 nodes
* Output layer: 3 nodes

## Dependencies

### Software Dependencies

<img src="https://github.com/kek01101/AC_computer_vision/blob/main/AC_CV/Picture2.png" height=80/>

Note: Anaconda is also required for downloading my python environment

### Game Mod Dependencies

I used the Mistubish Lancer EVOIII as my car for this project, it is a modded car for Assetto.
The mod used to add the Mitsubishi Lancer EVOIII to Assetto can be found [here](http://assetto-db.com/car/mitsubishi_lancer_evo3).

## Related Work

There are a couple of related projects which inspired me to implement computer vision into Assetto Corsa. Firstly, there was one of the showcased projects in either lecture 1 or 2, the one with the outdoor racing track. I thought that the idea of an autonomous racing car was a good one, but I do not have enough space to implement a real-world track like in the video: [https://github.com/rafah1/Robot-RACECAR](https://github.com/rafah1/Robot-RACECAR). Secondly, I was also inspired by this viral video of a genetic algorithm implemented into Trackmania, another well-known racing game: [https://youtu.be/a8Bo2DHrrow](https://youtu.be/a8Bo2DHrrow). While the approach for controlling the vehicle is entirely different, the main idea of implementing autonomy into a racing game is still there. Finally, I was inspired by the videos Roborace have uploaded to their YouTube channel of their alpha and beta seasons.

Overall, all of these different ideas combined to inspire me to implement an autonomous vehicle into Assetto Corsa, which as far as I can tell, has never been done before. In fact, when looking for resources online, one of the few similar projects I could find was a MatLab framework demonstration with 230 views on YouTube: [https://youtu.be/82morq4kVco](https://youtu.be/82morq4kVco). There is also another AI-based project for Assetto which takes a very similar approach to my project, but their writeup contains little code or other helpful resources for replicating their work: [https://devpost.com/software/self-driving-car-simulator-with-assetto-corsa](https://devpost.com/software/self-driving-car-simulator-with-assetto-corsa). Additionally, they never figured out how to feed their controls back to the car in Assetto, meaning that their vehicle was never able to drive. Essentially, this means that I had very few resources to build my project with, which made it a unique challenge.

## Results

### Baseline

To fully appreciate my results, it is best to first see a baseline, as dictated by my own driving. Note that all driving is in second gear, as the AI can not yet shift gears on its own.

START VIDEO GOES HERE

### 100 Epochs of Simple Training

This is the resulting driving after 100 epochs of simple training, this took around 20 minutes. Simple training means that the AI does not learn from its mistakes.

VIDEO GOES HERE

### 1000 Epochs of Custom Training

This is the resulting driving after 1000 epochs of custom training. The entire training took around 2 hours to complete, and 80 GB of checkpoint files were created. I decided not to push the checkpoint files to this repo. Custom training means that the AI does learn from its mistakes.

VIDEO GOES HERE

## Discussion/Conclusion

### Positive Aspects of this Project
I was quite surprised by how well the custom training model drove. Yes, it did not cross the lap line, but it was very close, and if I had just a bit more time it certainly would have. Additionally, the system managed to fulfill my objectives for this project, which is always nice.

### Negative Aspects of this Project
There were a couple of negative aspects to this project. First and foremost, for reasons described in my presentation, I did not have enough time to finish the computer vision aspect of my neural network. This is very dissapointing, and I wish there was more time to allow for me to finish that vital aspect of the project. Additionally, I was unhappy with how imprecise the keyboard controls were for the robot, and how it made training more difficult as a result. Finally, the amount of debugging I had to do in order for TensorFlow to even function was just far too high. At one point, I moved some functions and classes around within one of my scripts, and the entire script broke. Not only that, but I had to figure out what specific version numbers would work with my project, as apparently some version of TensorFlow are just completely broken.

### Future of this Project
There are many things I wish to add in the future for this project. Unsurprisingly, I would first like to add the promised computer vision component, as I feel that the project will be incomplete without it. Not only that, but I would also like to fix the precision issues which plague the AI as well. This will either be done by painstakingly creating my own python bridge for x360ce, or by finding a working one online. I would also like to increase the dataset size, as 10 runs is clearly not large enough. Creating 100 runs should take around 5 hours on the current map. 

In terms of far-future features I'd like to add to this project, there are also a couple. Firstly, I'd like to add more advanced racing techniques such as engine braking, as well as just adding gear shifting in general. With the current system, gear shifting would add too much complexity, and the playback function is not perfect enough just yet to handle such a dramatic shift. Eventually, I'd also like to race against real people with my AI, but that's VERY far in the future.

## References

* Obligatory "this project was powered by Tensorflow" - check them out [here](https://www.tensorflow.org).

# Code and Setup Information

## Code

### Neural Network Code
The neural network scripts can be found below, keep in mind that is important to use full paths when inputting directories, especially on linux machines

    training_final.py
    predicting_final.py

### Logging and Driving Code
The following scripts are used for logging inputs when manually driving, and for simulating inputs when letting the AI drive. Again, remember to use full directories.

    combined-driver.py
    combined-logger.py

Each script has a variable for a start_key, no logging or input simulating will be done until the start_key is pressed. It can be changed, but by default is the letter "p".

### Testing
All scripts in the testing folder are from prior testing and may not work correctly. Play around with them if you want. Be warned that the old input simulating scripts would freeze Assetto due to being too inefficient.

## Setup

### Environment Setup

If you would like to install my Python environment, first install the latest version of anaconda from [here](https://docs.anaconda.com/anaconda/install/windows/).

Next, download the tf.yml file from this repo. By default, you should put it in your User folder. 

Then, install the environment with the command:

    conda env create -f tf.yml

Accept all appropriate installation prompts, then activate the new environment with the command:

    conda activate tf

### Codebase Setup

Simply clone the repo into your work directory of choice, it's as easy as that!

## Issues

For any questions, please [create an issue](../..//issues).
