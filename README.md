# README
## TL;DR
`pip install -r requirements.txt`  
`python main.py`

## Further information
With the above commands the following jobs are executed: 
1. Data ingestion.
2. Data preparation.
3. Cluster modelling with algorithm comparison, metrics and model exports.
4. Regression modelling with algorithm comparison, metrics and model exports.

All visualizations and models are exported into the root directory and can be listed with the terminal command `ls` or `ll`. 
For quicker execution the calculations are conducted with a data sample of 10.000 records. 
The total amount of samples can be changed in `main.py.`. 
Moreover, the code for the data understanding is commented to speed up execution. It can also be included in the code flow by adjusting the `main.py`. 
