# srebp2-explanations

This is the repository for my summer research project consisting of applying Graph Neural Network (GNN) Explanation Methods to a GNN that is trained to predict whether a gene perturbation will have an effect on cellular and molecular phenotypes in the process of cellular cholesterol homeostasis. 

The files organization is as follows:
- adv-experiments: The advanced experiments folder includes all of the final experiments I conducted including:
    - Measure fidelity when edge mask is set to None to assess the importance of node features alone
        - Plot fidelity curves for various values of k (10, 20, 40, 80)
    - - Plot fidelity curves for subgraphs with various values of k (10, 20, 40, 80)
    - Revised stability calculations
        - Can we do 5  runs for each of 10  instances?
        - Calculate average Jaccard similarity for each instance
        - Make a box plot or something similar to show distribution of similarity measures across instances
- cora-example: The cora example folder includes a jupyter notebook that I used to explore the explanation methods with the Cora dataset example. 
- exploration: The exploration folder includes a puython notebook that I used to explore the NPC1 Node. 
- initial-experiment: The initial experiment folder includes a python notebook that I used to run initial experiments on a singular instance as a way to test before the advanced experiments.

# Advanced Experiments

The bulk of the work I did was in the advanced experiments folder. An overview of all of the code that I used can be found in the adv-experiments.ipynb file, which I used to create and test all of my code before running it in the .py files through the BMI server. 

## Nodes of Interest

- 9606.ENSP00000289989 lowest False Negative / predicted value: 0.0281679704785347 / node index: 17397
- 9606.ENSP00000358777 highest True Positive / predicted value: 0.9995648264884949 / node index: 9005
- 9606.ENSP00000216099 lowest True Negative / predicted value: 7.5380116e-30 / node index: 8665
- 9606.ENSP00000357637 highest False Positive / predicted value: 0.9998927 / node index: 12633
- 9606.ENSP00000357226 furthest to target with label 0 / predicted value: 1.4533093e-07 / node index: 18233
- 9606.ENSP00000344741 closest to target with label 1 / predicted value: 0.8364509 / node index: 12257
- 9606.ENSP00000470087 closest to target with label 0 / predicted value: 0.3407591 / node index: 1674
- 9606.ENSP00000289989 furthest to target with label 1 / predicted value: 0.02816797 / node index: 17397
- 9606.ENSP00000348069: SREBF1  / predicted value: 0.55870503 / node index: 6334
- 9606.ENSP00000370695: EXCO1 / predicted value: 0.50875914/ node index: 1883

The corresponding index can be found by the dictionary using the node_order.pickle document. 