from image_dataset import load_data
import torch.optim as optim	
from neural_network import NeuralNet
import torch.nn as nn

def run_classifier(folder_path, optimizer_string, workers, device):
	data_loader = load_data(250, workers, "train.csv",
							 folder_path, 32)

	neural_net = NeuralNet()

	criterion = nn.BCELoss()
	optimizer = get_optimizer(optimizer_string, neural_net)
	loss_outer_array = []
	precision1_outer_array = []
	precision3_outer_array = []
	for epoch in range(5):
		loss_array = []
		precision1_array = []
		precision3_array = []
		for i, data in enumerate(data_loader, 0):
			input_tensor, input_label_tensor = data['image_tensor'], data['image_label_tensor']
			optimizer.zero_grad()
			output_label_tensor = neural_net(input_tensor).float()
			precision = accuracy(output_label_tensor, input_label_tensor)
			loss = criterion(output_label_tensor, input_label_tensor)
			loss_array.append(round(loss.item(), 2))
			precision1_array.append(round(precision[0], 2))
			precision3_array.append(round(precision[1], 2))
			loss.backward()
			optimizer.step()
		loss_outer_array.append(loss_array)
		precision1_outer_array.append(precision1_array)
		precision3_outer_array.append(precision3_array)


	for epoch in range(5):
		for i, data in enumerate(data_loader, 0):
			print( "Epoch = {}, batch = {}, Loss = {}, Precision@1 {}, Precision@3 {}. \n"
				.format(epoch, i, loss_outer_array[epoch][i], precision1_outer_array[epoch][i], precision3_outer_array[epoch][i]))
	

	print("Training done")



def accuracy(output, target):
	one_correct = 0
	three_correct = 0
	for index in range(250):
		sample_output = output[index]
		sample_target = target[index]
		_, pred = sample_output.topk(3, 0)
		for k in range(3):
			if sample_target[pred[k]] == 1:
				if k == 0:
					one_correct = one_correct + 1

				three_correct = three_correct + 1

	return (round(one_correct/250, 2), round(three_correct/750, 2))
	

		
def get_optimizer(optimizer_string, neural_net):
	if optimizer_string == 'SGD':
		return optim.SGD(neural_net.parameters(), lr=0.01, momentum=0.9)
	if optimizer_string == 'SGDN':
		return optim.SGD(neural_net.parameters(), lr=0.01, momentum=0.9, nesterov=True)
	if optimizer_string == 'Adagrad':
		return optim.Adagrad(neural_net.parameters(), lr=0.01)		
	if optimizer_string == 'Adadelta':
		return optim.Adadelta(neural_net.parameters(), lr=0.01)
	if optimizer_string == 'Adam':
		return optim.Adam(neural_net.parameters(), lr=0.01, momentum=0.9)

