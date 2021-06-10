# sent2-cloud-remover

This project aim at first using the method of conditional Generative adversial Network on Sentinel 1 and Sentinel 2 
in order to be able to recreate Sentinel 2 images from previous Sentinel2 and 1 and current Sentinel 1.

- Training of the conditional Generative Adversial Network (done, some clean of the code required)
- Assessing  the results of the simulated image 
- Assessing the performance on changed area

## The environment : 
First you can use the specific training condo environment : env/training_env.yaml : 

```bash
conda env create -f env/training_env.yaml
conda activate training_env
python -m ipykernel install --user --name=training_env
```

### Start a training

#### With Jupyter

Start jupyter notebook. If in a remote machine : add  `--ip=0.0.0.0 --no-browser`

Now you can open the jupyter Notebook : notebooks/Trainings.ipynb

Modify the constant, defined at the beginning at the notebook and run the training. 

#### As a batch job

The cycle GAN model used is defined in models/clean_gan.py
In order to train a model two yaml should be modified, examples available in GAN_confs :  

- model.yaml
- train.yaml
  Then running gan_train.sh path_to_model_yaml path_to_train_yaml will start the training job

### Supervise the training

#### Tensorboard : metric supervision

The training is also configured to be supervised using Tensorboard. 

Open a new terminal window within training_env and run tensorboard.

Local command `tensorboard --logdir <path>`

If in remote machine : `tensorboard --logdir <path> --host 0.0.0.0`

#### Validation image visualization : notebook

Eventually you can also look at how the images look like during the training. 

Open the notebook *SPECIFY NOTEBOOK*


### Assessing the performance : 

The notebooks describing the results are available in ""









 



 
