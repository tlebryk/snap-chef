To run the pipeline: 

Run src/preparedataset.py pointing to the right locations. Try giving a GCP URI and troubleshoot. If that's not working, start locally to ensure functionality then build towards cloud integration. 

Next run train.py. I'd recommend from the previous step having a small dataset size so that train runs quickly and we can quickly see if the training crashes. Try everything up to the actual fit on CPU so you know the data loading is fine. 
Then, see if you can provision some GPU instance, SSH into it, and run this script with a small number of epochs. 

Finally, test inference.py to make sure we can run a trained adapter. 



