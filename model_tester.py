import tkinter as tk
from tkinter import filedialog
from params import TrainingParameters
import os
import glob
from agents import ContinuousDQN, LexCDQN
import multiprocessing
import gym



class TesterGUI(tk.Tk):

	def __init__(self):

		super().__init__()
		self.geometry("240x300")
		self.resizable(tk.FALSE, tk.FALSE)

		#GUI elements
		self.model_select = tk.Button(self, text = "Select Model", command = self.select_model)
		self.model_select.grid(row = 0, column = 0, padx = 10, pady = 10)

		self.info_label = tk.Label(self, text = "Model Info:")
		self.info_label.grid(row = 1, column = 0, padx = 10, pady = 10)
		
		self.infovar = tk.StringVar()
		self.infovar.set('NO INFO')		
		self.info_data = tk.Label(self,  textvariable = self.infovar)
		self.info_data.grid(row = 2, column = 0, padx = 10, pady = 10)

		self.render = tk.IntVar()
		self.render_check = tk.Checkbutton(self, text = "  Render Environment", variable = self.render)
		self.render_check.grid(row = 3, column = 0, padx = 10, pady = 10)

		self.infinite = tk.IntVar()
		self.infinite_check = tk.Checkbutton(self, text = "  Infinite Testing", variable = self.infinite)
		self.infinite_check.grid(row = 4, column = 0, padx = 10, pady = 10)

		self.test_model_butt = tk.Button(self, text = "Test Model", state = tk.DISABLED, command = self.test_thread)
		self.test_model_butt.grid(row = 5, column = 0, padx = 10, pady = 10)

		self.stop_test_butt = tk.Button(self, text = "Stop Testing", state = tk.DISABLED, command = self.stop_thread)
		self.stop_test_butt.grid(row = 6, column = 0, padx = 10, pady = 10)


		self.model_path = None
		self.model_params = None

		self.manager = multiprocessing.Manager()
		self.thread = None

		self.writer = None

	

	#Opens a filedialog to choose the model to test
	def select_model(self):

		params_path = None
		model_path = filedialog.askopenfile(title = "Choose a model", filetypes=[("Models", "*.pt")]).name

		model_dir = os.path.dirname(model_path)
		model_prev_dir = os.path.dirname(model_dir)

		#Search .params in model's directory
		for file in os.listdir(model_dir):

			if file.endswith('.params'):

				params_path = f'{model_dir}/{file}'
				break

		#If found, parse the file
		if params_path is not None:

			self.model_params = self.parse_dataclass(params_path, TrainingParameters)

		else: 

			#Search for .params file in the previous directory
			for file in os.listdir(model_prev_dir):

				if file.endswith('.params'):

					params_path = f'{model_prev_dir}/{file}'
					break

			if params_path is not None:

				self.model_params = self.parse_dataclass(params_path, TrainingParameters)
				# print("Params was found in the model previous directory")

			else:

				tk.messagebox.showwarning("Warning", "No parameters were found for the model")


		self.test_model_butt.config(state = tk.NORMAL)
		self.model_path = glob.glob(glob.escape(os.path.dirname(model_path)))[0]

		self.bake_infos()


	#GUI update function
	def update_fn(self):

		self.update()
		self.update_idletasks()

	#Generate an info string to display using labels, based on the .params file found
	def bake_infos(self):

		info_string = ''
		relevant_tags = [self.model_params.agent_name, self.model_params.bias, self.model_params.nohid, self.model_params.sample_size, 
							self.model_params.batch_size, self.model_params.hidden_size, self.model_params.num_episodes, self.model_params.slack,
							self.model_params.loss_threshold]

		info_string += 'CDQN' if relevant_tags[0] == "ContinuousDQN" else 'LexCDQN'
		info_string += ' [NOB]' if not relevant_tags[1]  else ''
		info_string += ' [NOHID]' if relevant_tags[2]  else ''
		if not relevant_tags[0] == "ContinuousDQN":
			info_string += f' [SL-{relevant_tags[7]*10}] [LT-{relevant_tags[8]*10}]'


		info_string += f' [{relevant_tags[3]}S]\n\n[{relevant_tags[5]},{relevant_tags[4]}]'


		self.infovar.set(info_string)

	
	#Start the thread for testing
	#>thread is necessary because otherwise the GUI would be blocked
	def test_thread(self):

		if self.thread == None:

			#Initialize agent
			model = LexCDQN if self.model_params.reward_size == 3 else ContinuousDQN	
			env = gym.make(self.model_params.env_name)
			agent = model(self.model_params, env, env.observation_space.shape[0], env.action_space.shape[0])
		
			#Initialize writer's path
			writer_path = os.path.join(self.model_path, 'tests')
			os.makedirs(writer_path, exist_ok=True)

			self.test_model_butt.config(state = tk.DISABLED)
			self.stop_test_butt.config(state = tk.NORMAL)
			
			#Start thread using the test classmethod
			self.thread = multiprocessing.Process(target=model.test, args = (agent, str(writer_path), self.model_path, 0, self.model_params, False, self.render.get(), self.infinite.get()))
			self.thread.start()	

	#Ends the test's thread, if any
	def stop_thread(self):

		if self.thread and self.thread.is_alive():

			self.thread.terminate()
			self.thread = None

			self.test_model_butt.config(state = tk.NORMAL)
			self.stop_test_butt.config(state = tk.DISABLED)

	#Retrieves dataclasses values starting from the lines read from the .params file
	def parse_dataclass(self, filename, dataclass):

		#Used to correctly initialize bool type values
		bool_mapping = {"true": True, "false": False}


		with open(filename, 'r') as f:

			lines = f.readlines()

		#Initialize standard dataclass
		params = dataclass()

		for line in lines[0].split('>,'):

			line = line.strip('< ')
			line = line.split(': ')
			
			#Check if line is empty
			if not line[0] == '':

				attr_name = line[0]
				attr_value = line[1]

				#Check if attribute value is valid
				if attr_value != 'None':

					#Cast attribute value to correct type and assign it to the standard dataclass
					if hasattr(dataclass, attr_name):

						attr_type = type(getattr(dataclass, attr_name))
						if attr_type is not bool:

							attr_value = attr_type(attr_value)

						else:

							attr_value = attr_value.lower()
							attr_value = bool_mapping[attr_value]

						setattr(dataclass, attr_name, attr_value)

		return dataclass






if __name__ == "__main__":

	win = TesterGUI()

	while True:

		win.update_fn()


