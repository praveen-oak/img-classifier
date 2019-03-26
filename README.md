This file contains all details of the structure of the codebase, how to run the classifier, and how to look at the results.

PLEASE NOTE : I have created multiple files instead of one big lab2.py file to improve code readibility.
Although it was mentioned in the assignment not to do this, subsequently the professor allowed multiple file
format in this piazza post

https://piazza.com/class/jr0t1qlu4wj3cj?cid=73




1. Running the code
There is a script called runner.sh in the main folder.
First make the script executable using the command:

	chmod 777 ./runner.sh

Then you just run the script

	./runner.sh

This should launch 11 runs on the HPC cluster.
You can check if the jobs have been launched using the command
squeue | grep <<your username here>>

Each job is named as:
	questionnumber_workers/optimizer

For eg:
If the job is running a task for C3, testing the code for 12 workers,
The job name would be c3_12

If the job is running a task for C3, testing the code for adagrad,
The job name would be c5_grad


2. Looking at the results
	The result files will be in the same main folder.
	All results files have the following name format

	questionnumber_workers/optimizer.out

	For example,
	If the result file is for question 3 with 16 workers, the generated result file will be named as:

		c3_12.out

	If the result file is for question 5 with adam optimizer, the generated result file will be named as:

		c5_adam.out


	After all the tasks have been finished, the following result files will be generated:
		
		c2_1.out -- Neural network with 1 worker, all precision numbers printed

		c3_0.out -- Neural network with 0 workers
		c3_4.out
		c3_8.out
		c3_16.out
		c3_20.out
		c3_24.out
		c3_28.out -- Neural network with 28 workers

		c5_adam.out -- Neural network run with adam optimizer
		c5_delta.out-- Neural network run with adadelta optimizer 
		c5_grad.out-- Neural network run with adagrad optimizer
		c5_sgd.out-- Neural network run with sgd optimizer
		c5_sgdn.out-- Neural network run with sgd nesterov optimizer


3. Result contents:

	**All timing data is in seconds**

	The result format is as follows:
	(Lines which will appear on output file is marked with <<>>)

	<<Running on CPU/Device>> -- Check this to make sure that the algo is running on the device requested

	If precision mode is turned on(For c2 and c5 parts)
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


