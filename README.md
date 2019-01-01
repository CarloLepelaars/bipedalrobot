# bipedalrobot
Training a virtual bipedal robot to walk in OpenAI Gym using Evolution Strategies (ES). 

Credit goes to [alirezamika](https://github.com/alirezamika/bipedal-es) for writing most of the code. I added documentation and optimized hyperparameters.

Feel free to tweak the hyperparameters in agent.py and see if you get a better reward.
The currently trained model gets a total reward of ±190.

Example video of one episode:

<a href="https://www.youtube.com/watch?v=iskHVlt0UBw
" target="_blank"><img src="https://i9.ytimg.com/vi/iskHVlt0UBw/mq2.jpg?sqp=CPjkkt4F&rs=AOn4CLDIW9pwFBsKufSMMQL0pz_wUgLjhg" 
alt="Video" width="300" height="180" border="10" /></a>

# Dependencies

To use this model you have to install [OpenAI Gym](https://github.com/openai/gym) and the [Evostra repository](https://github.com/alirezamika/evostra) of [alirezamika](https://github.com/alirezamika):

    pip install evostra
    pip install gym

# Training 
Type in command line:

    python agent_train.py
    
The program trains standard for 200 iterations. If you want you can change the amount of iterations in agent_train.py by changing the 'n' variable.

# Testing / Playing
Type in command line:

    python agent_play.py
    
The program plays one episode. If you want to change the number of episodes, change the 'n' variable in agent_play.py.

# Study material

This repository was created as a midterm assignment for School of AI's "Move 37" course.

For more information about the "Move 37" course check out:
https://www.theschool.ai/courses/move-37-course

For more information about Evolution Strategies (ES) in general:
https://blog.openai.com/evolution-strategies
