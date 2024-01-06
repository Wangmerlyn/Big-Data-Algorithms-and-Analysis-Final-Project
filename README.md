# Big-Data-Algorithms-and-Analysis-Final-Project

This repository contains the code for Cora and Citeseer dataset classfication.

---

Usage:
  python main.py [--dataset DATASET] [--model MODEL]

Options:
  --dataset DATASET       Specify the dataset to use. Possible values: 'Cora', 'CiteSeer'.
                          Default is 'Cora'.
  --model MODEL           Specify the model to use. Possible values: 'GFusion_1', 'VanillaGAT', 'VanillaGCN'.
                          Default is 'VanillaGAT'.

Examples:

  python main.py --dataset Cora --model VanillaGAT
  python main.py --dataset CiteSeer --model VanillaGCN

Additional Options:

  --epochs EPOCHS         Number of epochs to train. Default is 1001.

  --lr LR                 Initial learning rate. Default is 0.01.

  --weight_decay DECAY    Weight decay (L2 loss) coefficient. Default is 5e-4.


To run the model with custom settings, include these arguments in your command line. For example:
  python main.py --dataset Cora --model VanillaGAT --epochs 200 --lr 0.005 --weight_decay 1e-4

