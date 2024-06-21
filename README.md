# srebp2-explanations

This is the repository for my summer research project consisting of applying Graph Neural Network (GNN) Explanation Methods to a GNN that is trained to predict whether a gene perturbation will have an effect on cellular and molecular phenotypes in the process of cellular cholesterol homeostasis. 

The files are as follows: 

- node_order.pickle
- SimpleGAT_1_model_lr_0.0001_dp_0.7.pth
- gnn_explainer.ipynb

## Nodes of Interest

- '9606.ENSP00000269228' # NPC1 which is known to have involvement in the cholesterol homeostasis process
- '9606.ENSP00000289989' # Closest to SREBP2 and has label 1 (expected)
- '9606.ENSP00000415836' # Far from SREBP2 and has label 0 (expected)
- '9606.ENSP00000216180' # Closest to SREBP2 but has label 0 (unexpected)
- '9606.ENSP00000359398' # Far from SREBP2 but has label 1 (Unexpected)
- '9606.ENSP00000346046' # False Negative (lowest confidence)
- '9606.ENSP00000473036' # True Negative (lowest confidence)
- '9606.ENSP00000449270' # False Positive (highest confidence)
- '9606.ENSP00000270176' # True Positive (highest confidence)

The corresponding index can be found by the dictionary using the node_order.pickle document. 