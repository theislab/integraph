# integraph

## data
First, adapt the data paths in each notebook.<br>
Current structure of the data is as follows:.<br>
|data <br />
  |input <br />
    |complementary -> clinical variables are saved here <br />
    |original -> all six modalities in form of h5ad files <br />
|output <br />
  |analysis_input -> for each method of feature transformation, save relevant information <br />
    |maxabs <br />
    |minmax <br />
    |power <br />
    |quant_N <br />
    |quant_U <br />
    |standard <br />
    |wot -> The output of codes in notebooks predictions.ipynb and controls_and_contributions.ipynb are saved here, if no transformation was applied to the data <br />
    |raw -> cell counts <br />
  level1 <br />
  |level2 <br />
  |level3 <br />
  |shared_info_74 -> contains a map from IDs to Names and vice versa. Contains a map of IDs to pseudobulks. Contains the name of the 74 patients and their diseases <br />
       
## pseudobulk
*** This notebook should be the first to run! ***
- Generates pseudobulks in Level 1, Level 2 and Level 3 from the data. 
Data is stored in form of a dictionary, mapping ID to anndata object containing the pseudobulk without transformation. Transformed data is stored in layers of the anndata object.
- Extracts noise pseudobulks.
- Saves pseudobulks in the data folder of MoGCN as .csv files.
- Saves the output under Level 1/ Level 2/ Level 3

## psn
*** This notebook should be the second to run! ***
- Defines Penalize-Reward-score
- Performs hyperparameter optimization or uses standard hyperparameters
- Defines PSN weights for the wieghted average approach
- Generates PSNs using the hyperparameters and stores them as dictionary with key = Graph_ID and value = PSN (74x74 numpy arrays)
- Saves the output under Level 1/ Level 2/ Level 3

## controls_and_contributions
- Performs negative control: 1. Noise and Non-noise PSN are dissimilar, 2. Noise reduces prediction accuracy and the NMI acore with ground truth
- Performs positive control: 1. Similar PSNs (NMI above a threshold) have similar contributions to the final fused network (NMI difference whithin a threshold) 2. Average accuracy of all unimodal PSNs is smaller than the accuracy obtained after their fusion.
- Impements the majority vote approach for label prediction
- Computes the similarity of PSNs in different levels via Correlation 
- Saves the output under the name of the transformation applied to the data

## predictions
- Fuses networks, performs clustering on them, uses the majority vote approach to find labels.
- Uses accuracy, F1 score, Silouhette score and Panelize-Reward-score for evaluation
- Saves the output under the name of the transformation applied to the data

## analysis
*** This notebook should be the last to run! ***
- Computes relative cell counts
- Measures cluster quality using sillouhette score and NMI with ground truth (disease or disease group)
- Visualizes the distribution of different disease in each cluster
- performs gene ranking with wilcoxon test for every pseudobulk, grouping data by the clusters obtained from fusion methods, displaying 3 marker genes
- Compares disease and disease group accuracy and F1-score as well as penalize reward score for all appraoches, in all levels, with all transformations