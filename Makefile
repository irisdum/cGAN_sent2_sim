
source_directory=/srv/osirim/idumeur/data/dataset6/
target_directory=${source_directory}prepro1/
build_dataset_dir=${target_directory}build_dataset/

output_split_dir_name=input_large_dataset

path_train_yaml=GAN_confs/train.yaml
path_model_yaml=GAN_confs/model.yaml

training_dir=/srv/osirim/idumeur/trainings/new_model_corr/
training_number=42
pred_dataset=${target_directory}${output_split_dir_name}/test/
weight=295
pref_pred_image=tr${training_number}_w_${weight}_test_d6


train_model:
	python train.py --model_path ${path_model_yaml} --train_path ${path_model_yaml}

train_on_cluster:
	oarsub -q production "nodes=1/gpu=2,walltime=25:00:00" -p "cluster='grimani'" ./gri5000_train.sh ${path_model_yaml} ${path_model_yaml} ${target_directory}${output_split_dir_name}

predict:
	sbatch gan_predict.sh ${training_dir} ${training_number} ${weight} ${pred_dataset} ${pref_pred_image} ${source_directory}

predict_val:
	sbatch gan_predict_val.sh ${training_dir} ${training_number} ${weight}

