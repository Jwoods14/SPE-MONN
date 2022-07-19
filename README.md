# SPE-MONN
### Requirements:
Python 2.7  
RDkit  
Pytorch >= 0.4.0  
Scikit-learn  
### Usage:  
1 The benchmark dataset can be found in ./data, and the data processing can follow protocol in  **[MONN: a Multi-Objective Neural Network for Predicting Pairwise Non-Covalent Interactions and Binding Affinities between Compounds and Proteins](https://github.com/lishuya17/MONN)**.  
The PtsRep protein embeddings can be found in our another article **[Self-Supervised Representation Learning of Protein Tertiary Structures (PtsRep) and Its Implications for Protein Engineering](https://www.biorxiv.org/content/10.1101/2020.12.22.423916v2.abstract)**, and TAPE or UniRep can be found in this **[repo](https://github.com/songlab-cal/tape)**.

2 For training and testing your model, you can use the `train_test.py`and run with following command on KIKD dataset, new-compound setting, clustering threshold 0.3:  
`  python train_test.py KIKD new_compound 0.3  `.  
3 For testing model on some compounds and proteins with trained model, you can use the `mol_test.py`.
