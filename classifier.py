from image_dataset import load_data
import torch.optim as optim	
from neural_network import NeuralNet
import torch.nn as nn

def run_classifier():
	data_loader = load_data(250, 1, "train.csv",
							 "/scratch/gd66/spring2019/lab2/kaggleamazon/", 32)

	neural_net = NeuralNet()

	criterion = nn.BCELoss()
	optimizer = optim.SGD(neural_net.parameters(), lr=0.01, momentum=0.9)

	for epoch in range(5):
		for i, data in enumerate(data_loader, 0):
			input_tensor, input_label_tensor = data['image_tensor'], data['image_label_tensor']
			# print(image_tensor)
			# break
			optimizer.zero_grad()
			output_label_tensor = neural_net(input_tensor).float()
			loss = criterion(output_label_tensor, input_label_tensor)
			loss.backward()
			optimizer.step()

	print("Training done")

run_classifier()

