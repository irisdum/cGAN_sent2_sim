# sent2-cloud-remover
These software codes can be used for non-profit academic research only.
They are distributed under the terms of the GNU general public license v3.

Code corresponding to **INSERT PAPER NAME*

## The environment : 
Use the specific training conda environment : env/training_env.yaml : 

```bash
conda env create -f env/training_env.yaml
conda activate training_env
python -m ipykernel install --user --name=training_env
```

### Start a training

#### With Jupyter

Start jupyter notebook. If in a remote machine : add  `--ip=0.0.0.0 --no-browser`

You can open the jupyter Notebook : notebooks/Trainings.ipynb

Modify the constant, defined at the beginning at the notebook and run the training. 

#### As a batch job

The conditional GAN model used is defined in models/clean_gan.py
In order to train a model two yaml should be modified, examples available in GAN_confs :  

- model.yaml
- train.yaml
Then run 
```gan_train.sh path_to_model_yaml path_to_train_yaml ``` 

### Supervise the training

#### Tensorboard : metric supervision

The training is also configured to be supervised using Tensorboard. 

Open a new terminal window within training_env and run tensorboard.

Local command `tensorboard --logdir <path>`

If in remote machine : `tensorboard --logdir <path> --host 0.0.0.0`

#### Validation image visualization : notebook

Eventually you can also look at how the images look like during the training. 

Open the notebook Visualize_training_data


### Assessing the performance : 

The notebooks describing the results are available in ""

### Prediction

To run prediction of the model. You can run in python
`predict.py --model_path ${model_path}   --tr_nber ${train_nber} --dataset ${dataset} --pref ${pref} --weights ${weight}`



### Compute metrics

Open Notebooks Assessment_cGAN-CD





 



 
