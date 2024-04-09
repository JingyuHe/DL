# Demonstration
An example for the main function part for the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3243683)

#### Data
Three files in "data" folder:   
-- 60chars.parquet  
-- ff5chars.parquet  
-- winsorized_returns.parquet  

#### Parameter Setting
n_stocks = 3000  
In-Sample: 24 months (72000 observations)  
Out-of-Sample: 6 months (18000 observations)  
Do cross-validation to choose the best MSE as parameters combination;  
Epoch = 100  
Batch = 120  
layer_list = [0, 1, 2, 3]  
g_dim_list = [1, 5]  
n_factor = 5  
    
#### Non-linearbeta Setting
-- Use "tanh" activation function for beta layer;  
-- Use 60 characteristics to build up beta hidden layers (n_beta = 60);  
-- l1_v_list = [-4, -5, -6];  
-- l1_beta = l1_v_list - 6;  
-- beta_hidden_sizes = [60, 16, 4];  
-- Random Seed = 50;  
    
