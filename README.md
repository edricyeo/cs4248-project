### Python Scripts

`app.py`: Training script with hyper-parameter search.

`inference.py`: Running inference on trained models to get answer files saved in `answers` folder.

`answers`: Folder containing all answers and evaluation files

`eval_0.json`: Answers with the highest scores, corresponding to trial 0

### Instructions to run training and inference.

1) Create a conda environment with the given `environment.yml`
2) Run the `train.sh` script to submit SLURM job. This runs hyper-parameter search for 10 trials on `XLNet-large-cased`.
    - Note that to run inference for each of the trials, add the model path to the list of paths at the top of the `inference.py` file.
3) After setting model paths, run the `eval.sh` script. This then produces the respective answer files and evaluation results.

### Instructions for view Training Plots

Ensure you have `tensorboard` installed and run
```
tensorboard --logdir=logs
```