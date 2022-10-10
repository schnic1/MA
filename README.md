Welcome!

This repository is the code and tutorial to the master thesis "Reinforcement Learning for Optimal Trading Rule Selection" at the University of Zurich.

Please read this file carefully as it serves as a manual for the implemented reinforcement learning algorithm.

To run the code you will need the all the exact libraries as in requirements.txt, otherwise the program might not work properly.

Additional information:
- The program will recreate the folder structure if any folder is missing. It will not overwrite existing folders.
- If the folder already existed before running the program, the generated files will be added to the existing folder.
- Newly generated files will overwrite existing files with the same name. 
- For training a new agent, it is best to delete the 'model_data' folder or save it elsewhere.
- The results of each episode will be printed to the console.


Please note, that all adjustments only need to be done in the "config.py" file and running the code always refers to "main.py". 

### Train a new agent
1. Indicate what agent algorithm you want to train by setting the ```method``` variable to either ```'a2c'``` or ```'ppo'```.
2. Set ```run_training``` to ```True```.
3. Set ```evaluation``` to ```False```.
4. You can redefine the ```TOTAL_TIME_STEPS``` variable to be any integer. It defines the total training steps of the agent.
5. You can adjust any values in the ```AGENT_PARAM_DICT``` dictionary any value within their allowed range.
6. Indicate what reward function you want to use by setting ```'reward_arg'``` to either ```'return'``` or ```'sharpe'``` in the ```env_kwargs``` dictionary.
7. Run the program.
8. The models are saved to the 'model_data/models' folder.
9. The output files are saved to the respective folder in 'model_data/episode_data'.
10. The evaluation will not happen automatically, as 

### Retest a trained agent
1. Make sure to delete the 'model_data/episode_data' folder or save it elsewhere.
2. Indicate what agent algorithm you want to test, by setting the ```method``` variable to either ```'a2c'``` or ```'ppo'```.
3. Set ```run_training``` to ```False```.
4. Set ```evaluation``` to ```False```.
5. Choose the trained agent in the 'model_data/models' folder and set ```trained_model``` to the name of the model. (e.g. ```trained_model``` = ```'PPO_model_5.zip'```)
6. Make sure that ```'reward_arg'``` in the ```env_kwargs``` dictionary is set to the reward function, the agent was trained on.
7. Run the program.
8. The output files are saved to the respective folder in 'model_data/episode_data'.


### Evaluate the model performance
1. Set the ```run_training``` to ```False```.
2. Set the ```evaluation``` variable to ```True```.
3. You can adjust the benchmark's fixed positions to other values.
4. Choose the episode to evaluate in the 'model_data/episode_data/test_pred' folder and set ```evaluation_period``` to the name of the episode. (e.g. ```evaluation_period``` = ```'PPO_model_5.zip'```)
5. Run the program.
6. The output files are saved to the 'model_data/evaluation' folder.
