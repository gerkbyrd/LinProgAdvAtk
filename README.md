# Example Commands
After downloading the repository, these are to be run from the root folder of the repository. The code assumes a GPU with CUDA is available.
TODO: Add file with dependencies.
For now everything uses CIFAR-10 and the CIFAR-Large model, the code handles the automatic download and pre-processing of CIFAR-10.
IMPORTANT: the paper does not say how points are in the rectangular mesh that interpolates the range of interest. Here we assume 20, using the parameter interpol=meshpoints/2

## To train robust network using EMRobust with default params run:
python main.py --method em --epsilon 2/255 --alpha 0.95 --interpol 10

## To train robust network using Madry (does PGD attacks during training instead of easily misclassified examples) run:
python main.py --method madry --epsilon 2/255
