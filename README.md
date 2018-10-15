# bipedalrobot
Training a virtual bipedal robot to walk in OpenAI Gym using Evolution Strategies (ES)

Creating for the model goes to [alirezamika](https://github.com/alirezamika/bipedal-es) for most of the code. I just added some documentation and optimezed a few hyperparameters.

Feel free to tweak the hyperparameters in agent.py and see if you get a better reward.
The currently trained model has a total reward of ±190.

Example video of one episode:

<a href="https://www.youtube.com/watch?v=iskHVlt0UBw
" target="_blank"><img src="https://i9.ytimg.com/vi/iskHVlt0UBw/mq2.jpg?sqp=CPjkkt4F&rs=AOn4CLDIW9pwFBsKufSMMQL0pz_wUgLjhg" 
alt="IMAGE ALT TEXT HERE" width="300" height="180" border="10" /></a>

# Training 
Type in command line:

    python agent_train.py
    
The program trains standard for 200 iterations. If you want you can change the amount of iteration in agent_train.py by changing the 'n' variable.

# Testing / Playing
Type in command line:

    python agent_play.py
    
The program plays one episode as a standard. If you want to change the number of episodes, change the 'n' variable in agent_play.py


This repository was created as a midterm assignment for School of AI's "Move 37" course.

For more information check out:
https://www.theschool.ai/courses/move-37-course
