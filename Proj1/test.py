from metrics import Cross_validation
from architecture import LugiaNet, oO_Net, Naive_net
import time

max_epochs = 30
granularity = 2
runs = 15

# List of the architectures you want to assess
# First only architectures with hyperparameters
architectures = [LugiaNet,LugiaNet,LugiaNet,oO_Net]
# List of the arguments for each architecture
args = [[2],[3],[4],[5]]
# Initialize the cross validation to determine the hyperparameters of some architectures
validation_algo = Cross_validation(architectures,args,epochs=max_epochs,steps=granularity,runs=runs)

validation_algo.run_all()
validation_algo.plot_full_comparison(save_folder="figures/")

print("\n \n The graph with the full comparison has been saved in the figures folder, select the best hyperparameters please")
print("Wait... (we need time to choose the best ones (done manually))")
time.sleep(5)
print("Ok we determined the best hyperparameters, let's assess the performances on the test set \n \n")

# Architectures to test on the test set 
final_architectures = [LugiaNet,LugiaNet,LugiaNet,oO_Net]
# List of the best hyperparameters found so far
args = [[2],[3],[4],[5]]
# Initialize the cross validation to determine the hyperparameters of some architectures
Test_algo = Cross_validation(architectures,args,epochs=max_epochs,steps=granularity,runs=runs)

validation_algo.run_all(test=True)
validation_algo.plot_full_comparison(save_folder="figures/",test=True)

print("The final results are available in the figures folder")