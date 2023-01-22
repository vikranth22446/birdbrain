# Birdbrain: Analyzing Waterbirds and Mobilenet
Simply running metrics such as model information capacity, signal to noise ratio, and others via last layer retraining. 

Generate the csvs for brainome via:
```
python3 compute_memorization_csv.py
```

To run the experiment with those csvs:
    - different SNR values
    - training loss curves
    - waterbird subgroup accuracy
    - validation test curves
    - different capacities/sizes of hidden layers

```
python3 run_compressed_csv_experiments.py
```

The experimental results are stored in the experimental results folder. The graphs can be generated via running the cells of `plot_notebook.ipynb`


