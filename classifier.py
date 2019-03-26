from image_dataset import load_data
import torch.optim as optim	
from neural_network import NeuralNet
import torch.nn as nn
import time


#method takes in the command line parameters, creates the data loader and the neural network
#and then runs the data on the neural network, collects the required stats and prints them to file
def run_classifier(folder_path, optimizer_string, workers, device, calculate_precision=False):
	data_loader = load_data(250, workers, "train.csv",
							 folder_path, 32)

	data_load_time = 0
	neural_net = NeuralNet().to(device)
	criterion = nn.BCELoss()
	optimizer = get_optimizer(optimizer_string, neural_net)
	loss_outer_array = []
	precision1_outer_array = []
	precision3_outer_array = []

	epoch_start_time = time.monotonic()
	for epoch in range(5):
		if calculate_precision:
			loss_array = []
			precision1_array = []
			precision3_array = []

		enumerator = enumerate(data_loader)

		data_load_start_time = time.monotonic()
		compute_start_time = data_load_start_time
		data_load_time = 0
		compute_time = 0

		for i, data in enumerator:
			data_load_end_time = time.monotonic()
			data_load_time = data_load_time + (data_load_end_time - data_load_start_time)

			input_tensor, input_label_tensor = data['image_tensor'].to(device), data['image_label_tensor'].to(device)
			optimizer.zero_grad()

			output_label_tensor = neural_net(input_tensor).float()
			loss = criterion(output_label_tensor, input_label_tensor)
			loss.backward()
			optimizer.step()
			compute_end_time = time.monotonic()
			compute_time = compute_time + (compute_end_time - compute_start_time)

			if calculate_precision:
				precision = accuracy(output_label_tensor, input_label_tensor)
				precision1_array.append(round(precision[0], 2))
				precision3_array.append(round(precision[1], 2))
				loss_array.append(round(loss.item(), 2))

			compute_start_time = time.monotonic()
			data_load_start_time = time.monotonic()

		if calculate_precision:
			loss_outer_array.append(loss_array)
			precision1_outer_array.append(precision1_array)
			precision3_outer_array.append(precision3_array)

	epoch_end_time = time.monotonic()

	epoch_time = epoch_end_time - epoch_start_time

	print("-----------------Training done------------------------")

	if calculate_precision:
		print("-----------------Printing precision stats------------------------")
		for epoch in range(5):
			for i, data in enumerate(data_loader, 0):
				print( "Epoch = {}, batch = {}, Loss = {}, Precision@1 {}, Precision@3 {}. \n"
					.format(epoch, i, loss_outer_array[epoch][i], precision1_outer_array[epoch][i], precision3_outer_array[epoch][i]))
		
		average_loss = sum(sum(x) for x in loss_outer_array)/600.0
		average_precision_1 = sum(sum(x) for x in precision1_outer_array)/600.0
		average_precision_3 = sum(sum(x) for x in precision3_outer_array)/600.0

		print("Average loss over 5 epoch = {}".format(average_loss))
		print("Average precision@1 over 5 epoch = {}".format(average_precision_1))
		print("Average precision@3 over 5 epoch = {}".format(average_precision_3))



	print("-----------------Printing aggregate stats------------------------")
	print("Aggregated time for data loading = {}".format(data_load_time))
	print("Aggregated time for mini-batch computation = {}".format(compute_time))
	print("Aggregated time for each epoch = {}".format(epoch_time))



	print("-----------------Printing average stats------------------------")
	print("Average time per mini batch for data loading = {}".format(round(data_load_time/600.0, 2)))
	print("Average time per minibatch for computation = {}".format(round(compute_time/600.0, 2)))
	print("Average time for each epoch = {}".format(round(epoch_time/5.0, 2)))




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
		return optim.Adam(neural_net.parameters(), lr=0.01)

