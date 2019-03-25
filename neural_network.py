import torch
import torch.nn as nn
import torch.nn.functional as functional

class NeuralNet(nn.Module):
	def __init__(self):
		super(NeuralNet, self).__init__()
		#imput channel, output channel, 
		self.inception1_conv1 = nn.Conv2d(3, 10, 1)
		self.inception1_conv3 = nn.Conv2d(3, 10, 3, padding=1)
		self.inception1_conv5 = nn.Conv2d(3, 10, 5, padding=2)


		self.inception2_conv1 = nn.Conv2d(30, 10, 1)
		self.inception2_conv3 = nn.Conv2d(30, 10, 3, padding=1)
		self.inception2_conv5 = nn.Conv2d(30, 10, 5, padding=2)

		#for fc1, input_tensor is 30*8*8 tensor which is changed to 1*1920 tensor

		self.fc1 = nn.Linear(1920, 256)
		self.fc2 = nn.Linear(256, 17)

	def forward(self, input_tensor):
		input_tensor = self.first_inception(input_tensor)
		input_tensor = self.second_inception(input_tensor)
		input_tensor = self.fully_connected(input_tensor)
		return input_tensor

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

	def fully_connected(self, input_tensor):
		#first convert the tensor to a 1D tenson
		input_tensor = input_tensor.view(-1, self.num_flat_features(input_tensor))
		#fc1 then run relu
		input_tensor = functional.relu(self.fc1(input_tensor))
		#fc2 then run sigmoid
		input_tensor = torch.sigmoid(self.fc2(input_tensor))
		return input_tensor

	def first_inception(self, input_tensor):
		output_1 = self.inception1_conv1(input_tensor)
		output_3 = self.inception1_conv3(input_tensor)
		output_5 = self.inception1_conv5(input_tensor)

		#now concatanate the three convolutions along the channel, which is at the 1st index of tensor
		concat_output = torch.cat((output_1, output_3, output_5), 1)
		relu_output = functional.relu(concat_output)
		max_pool_output = functional.max_pool2d(relu_output, 2)
		return max_pool_output

	def second_inception(self, input_tensor):
		output_1 = self.inception2_conv1(input_tensor)
		output_3 = self.inception2_conv3(input_tensor)
		output_5 = self.inception2_conv5(input_tensor)

		#now concatanate the three convolutions along the channel, which is at the 1st index of tensor
		concat_output = torch.cat((output_1, output_3, output_5), 1)
		relu_output = functional.relu(concat_output)
		max_pool_output = functional.max_pool2d(relu_output, 2)
		return max_pool_output


