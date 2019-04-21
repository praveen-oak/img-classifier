This file contains all details of the structure of the codebase, how to run the classifier, and how to look at the results.

1. Running the code
There is a script called runner.sh in the main folder.
First make the script executable using the command:

	chmod 777 ./runner.sh

Then you just run the script

	./runner.sh

This should launch 11 runs on the HPC cluster.

2. Looking at the results
	The result files will be in the same main folder.

3. Result contents:

	**All timing data is in seconds**

	The result format is as follows:
	(Lines which will appear on output file is marked with <<>>)

	<<Running on CPU/Device>> -- Check this to make sure that the algo is running on the device requested

	If precision mode is turned on,
	Next 600 lines will be as follows
	<<Epoch = 0, batch = 29, Loss = 0.43, Precision@1 0.9, Precision@3 0.63. >>
		Each line reporting the batch epoch along with the corresponding loss and precision numbers


	<<Training done>>
	<<-----------------Printing aggregate stats------------------------>>
	<<Aggregated time for data loading = 8.038120353245176>>
	<<Aggregated time for mini-batch computation = 91.46839313209057>>
	<<Aggregated time for each epoch = 452.6495665649418>>
	<<-----------------Printing average stats------------------------>>
	<<Average time per mini batch for data loading = 0.01>>
	<<Average time per minibatch for computation = 0.15>>
	<<Average time for each epoch = 90.53>>


	Here aggregate time is one that is summed over all runs and average time is aggregate/5 for epoch and aggregate/600 for data loading and mini-batch


4. Structure of code base
	The code is divided into 4 files
	1. main.py - Called from the command line, picks up arguments and starts the training process
	2. image_dataset.py - Responsible for building the data loader as well as manipulating the image and label to a tensor and cropping and resizing the image as required
	3. neural_network.py - Builds the entire CNN as mentioned in the assignment requirements
	4. image_dataset.py - Creates the dataloader, the neural net and the trains it. Collectes the time and accuracy information and prints it out to the file.

	Other than the code there are a few helper file and folders to help with creating the running the slurm jobs
	1. ./runner.sh - Script to run all the jobs
	2. launcher folder - Contains the sbatch profile files for each question.


