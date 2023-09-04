import inspect


from dataclasses import dataclass


@dataclass
class TrainingParameters:

	gamma: float = 0.99
	learning_rate: float = 1e-3

	epsilon: float  = 0.9
	epsilon_decay: float = 0.99
	epsilon_min: float = 0.0
	epsilon_decay_start: int = 10

	slack: float = 0.001
	loss_threshold: float = 0.5

	update_every: int = 4
	save_every_n: int = 20
	save_path_every_n: int = 200
	save_path_every_n_test: int = 10

	batch_size: int = 4
	buffer_size: int = int(1e4)
	sample_size: int = 20
	hidden_size: int = 256
	reward_size: int = 1

	nohid: bool = False
	bias: bool = True

	env_name: str = "None"
	agent_name: str = "None"

	num_test: int = 200
	num_episodes: int = 1000

# After dataclass attributes are initialised, validate the training parameters
	def render_and_print(self):
		print(self.render_to_string())

	def render_to_string(self):
		x = ""
		for atr_name, atr in inspect.getmembers(self):
			if not atr_name.startswith("_") and not inspect.ismethod(atr):
				x += f" < {atr_name}: {str(atr)} >, "
		return x

	def render_to_file(self, dir):
		x = self.render_to_string()
		with open(dir, "w") as f: f.write(x)
