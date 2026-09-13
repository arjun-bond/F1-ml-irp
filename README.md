# Arjun F1 Machine Learning IRP source code

````
# F1 ML IRP

## Reproducing the Results

To reproduce the results presented in the research paper, run the following scripts **in order**:

bash
python collect_data.py
python build_dataset.py
python CalculateStats.py
python train_model.py
python evaluate_model.py
````

 > **Note:** Each script depends on the output of the previous step, so the scripts must be run in the order shown above.

 ## FastF1 Cache

 To calculate the size of the FastF1 cache, run:

```
python cache_size.py
```
