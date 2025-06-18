class ModelWrapper:

	def __init__(self, model_name, model, device):

		self.name = model_name
		self.model = model
		self.device = device

	def predict(self, batch_tensor):

		if self.name == 'pointnet':

			outputs, _ = self.model(batch_tensor)
			return outputs

		elif self.name == 'pointnetpp':

			outputs = self.model(batch_tensor)
			return outputs

		else:

			raise ValueError(f"Unknown model name: {self.name}")
