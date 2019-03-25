import argparse
from classifier import run_classifier
import torch


def main():
	available_optimizers = ['SGD', 'SGDN', 'Adagrad', 'Adadelta', 'Adam']
	parser = argparse.ArgumentParser(description='Image classifier using basic version of inception net')
	parser.add_argument('-o','--optim', help='Select optimizer from SGD, SGDN, Adagrad, Adadelta, Adam.'
		, required=True, choices=available_optimizers)
	devices = ['CPU', 'GPU']
	parser.add_argument('-c','--device', help='Select CPU or CPU. Default CPU', choices=devices)
	parser.add_argument('-d','--data_path', help='Folder where images and labels are stored', required=True)
	parser.add_argument('-w','--workers', help='Number of workers, default 1')
	parser.add_argument('-p','--precision', help='T = Report precison numbers, anything else not reported. Default not reported')

	args = vars(parser.parse_args())
	folder_path = args['data_path']
	optimizer_string = args['optim']
	workers = None
	device = None
	if args['workers'] == None:
		workers = 1
	else:
		workers = int(args['workers'])
	if args['device'] != None:
		if torch.cuda.is_available() == False:
			exit("No GPU available")
		else:
			device = torch.device("cuda")
			print("Running on device : {}".format(torch.cuda.is_available()))
	else:
		device = torch.device("cpu")
		print("Running on the CPU")

	if workers == 1:
		run_classifier(folder_path, optimizer_string, workers, device, calculate_precision=True)
	else:
		run_classifier(folder_path, optimizer_string, workers, device)

if __name__ == '__main__':
	main()
	