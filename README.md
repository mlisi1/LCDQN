# Lexicographyc Continuous Deep Q-Network (LCDQN)
This repository presents the code for the project Lexicographic Reinforcement Learning
Benchmark on MuJoCo environment, realized for the Symbolic and Evolutionary Artificial Inteligence course in University of Pisa.\
The goal of this project is to create a benchmark to see the advantages of Lexicographic Multi-Objective Reinforcement Learning (LMORL) over classic Reinforcement Learning (RL).
The main difference in these two approaches is that, while RL is inherently a single objective optimization problem, LMORL has the possibility to optimize multiple objectives hierarchically ordered based on their importance.\
The project was realized in Python, mainly using `PyTorch`, `numpy` and `Gym`, particularly the MuJoCo Ant Environment.
![Ant Environment](https://www.gymlibrary.dev/_images/ant.gif)

# Usage
## Training
All the trainings can be started by `train.py` and are automated based on the variables specified in the file:
- **hidden**: size of the intermediate layers of the network
- **batch**: batch size for the Replay Buffer
- **tags**: string tags to select the type of reward/penality (the `compute_rewards` function in `agents.py` needs to be modified accordingly)
- **descriptor_base**: string tags describing the model
- **env**: the Gym environment
- **params**: a `TrainingParameters` dataclass defining all the models information

The training will happen for all the possible combinations of reward types and intermediate layers dimensions and will be saved in the `trainings` folder in a folder named after the tags used.\
Along with the network's weights for the best performing model, and the average best performing model, a .ini file with the TrainingParameters data will be saved, as well as a `tfevents` binary file that can be opened by **TensorBoard**.
As an alternative, [tf_reader](https://github.com/mlisi1/tf_reader) has been developed specifically to open the training results and plot relevant data exploiting the tags system.

## Test
The file `model_tester.py` will open a dialog box letting you choose the .pt file containing the model, and will automatically load its relative infos and run a simuation to see the agent in action.

# Further Information
If needed `Report.pdf` and `Appendix.pdf` contain respectively the report describing the project and its development, and the plots relative to the training of all the models.
