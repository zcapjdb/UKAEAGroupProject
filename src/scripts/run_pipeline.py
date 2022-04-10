# Load the required data
import os 
from scripts.pipeline_tools import (
    prepare_data,
    regressor_uncertainty,
    select_unstable_data,
    retrain_regressor,
    retrain_classifier,
    uncertainty_change,
    mse_change,
)
from scripts.Models import ITGDatasetDF, load_model, ITGDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle

import coloredlogs, verboselogs, logging


level = "DEBUG"
# Create logger object for use in pipeline
verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level=level)

# Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50


with open('/home/tmadula/UKAEAGroupProject/src/scripts/pipeline_config.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

pretrained = cfg['pretrained']
paths = cfg['data']
save_paths = cfg['save_paths']

# RETRAIN_CLASSIFIER = False

train_data, val_data = prepare_data(
    paths['train'], paths['validation'], target_column="efiitg_gb", target_var="itg"
)

scaler = StandardScaler()
scaler.fit_transform(train_data.drop(["itg"], axis=1))

train_dataset = ITGDatasetDF(train_data, target_column="efiitg_gb", target_var="itg")
valid_dataset = ITGDatasetDF(val_data, target_column="efiitg_gb", target_var="itg")

# # TODO: further testing of the scale function
train_dataset.scale(scaler)
valid_dataset.scale(scaler)

# Load pretrained models
print("Loaded the following models:\n")
models = {}
for model in pretrained:
    if pretrained[model]["trained"] == True:
        trained_model = load_model(model, pretrained[model]["save_path"])
        models[model] = trained_model

# Train untrained models (may not be needed)

# Sample subset of data to use in active learning (10K for now)
# TODO: Needs to be the true training samples used!!!
train_sample = train_dataset.sample(10_000)

# # Pass points through the ITG Classifier and return points that pass (what threshold?)
# valid_sample, misclassified_sample = select_unstable_data(valid_sample, batch_size=100, classifier=models["ITG_class"])

# if RETRAIN_CLASSIFIER == True:
#     # retrain the classifier on the misclassified points
#     train_loss, train_acc, val_loss, val_acc = retrain_classifier(
#         misclassified_sample,
#         valid_dataset,
#         models["ITG_class"],
#         batch_size=100,
#         epochs=5,
#         verbose=True,
    # )
#TODO: diagnose how well the classifier retraining does
#TODO: verbose flag a good way to control the amount of output from different functins - not yet implemented
# From first run through it does seem like training on the misclassified points hurts the validation dataset accuracy quite a bit

# should be added to config file 
lam = 0.0
init_epoch = 50
iterations = 5

train_losses = []
test_losses = []
n_train_points = []
mse_before = []
mse_after = []
d_mse = []
d_train_uncert = []

for i in range(iterations):
    print(f"\nIteration: {i}\n")
    valid_sample = valid_dataset.sample(10_000)

    # remove the sampled data points from the dataset
    valid_dataset.remove(valid_sample.data.index)
    
    select_unstable_data(valid_sample, batch_size=100, classifier=models["ITG_class"])
    
    uncertain_datset, uncert_before, data_idx = regressor_uncertainty(
    valid_sample, models["ITG_reg"], n_runs=15, keep=0.25,valid_dataset=valid_dataset
)
    train_sample_origin, train_uncert_before, train_uncert_idx = regressor_uncertainty(
        train_sample, models["ITG_reg"], n_runs=15,train_data=True,
        )

    train_sample.add(uncertain_datset)

    uncertain_loader = DataLoader(train_sample, batch_size=len(train_sample), shuffle=True)
    
    prediction_before, prediction_idx_order = models["ITG_reg"].predict(uncertain_loader)

    # Switching validation dataset to numpy arrays to see if it is quicker
    x_array = valid_dataset.data[train_keys].values
    y_array = valid_dataset.data["itg"].values
    z_array = valid_dataset.data["efiitg_gb"].values
    dataset_numpy = ITGDataset(x_array, y_array, z_array)
    valid_loader = DataLoader(
        dataset_numpy, batch_size=int(0.1 * len(y_array)), shuffle=True
    )

    # Retrain Regressor (Further research required)
    epochs = init_epoch * (i+1)
    train_loss, test_loss = retrain_regressor(
        uncertain_loader,
        valid_loader,
        models["ITG_reg"],
        learning_rate=1e-3,
        epochs=epochs,
        validation_step=True,
        lam = 0.6
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    prediction_after,_ = models["ITG_reg"].predict(uncertain_loader, prediction_idx_order)

    _, uncert_after,_ = regressor_uncertainty(valid_sample, models["ITG_reg"], n_runs=15, keep=0.25, order_idx=data_idx)
    _, train_uncert_after,_ = regressor_uncertainty(train_sample_origin, models["ITG_reg"], n_runs=15,order_idx=train_uncert_idx, train_data=True)
   
    _ = uncertainty_change(x=uncert_before, y=uncert_after)
    
    d_train_uncert.append(uncertainty_change(x=train_uncert_before, y=train_uncert_after))
    
    _ = mse_change(prediction_before, prediction_after,prediction_idx_order,data_idx, uncertain_loader,[uncert_before, uncert_after])

    train_mse_before, train_mse_after, delta_mse = mse_change(
    prediction_before,
    prediction_after,
    prediction_idx_order,
    train_uncert_idx,
    uncertain_loader,
    uncertainties=[train_uncert_before, train_uncert_after],
    data="train"
     )
    mse_before.append(train_mse_before)
    mse_after.append(train_mse_after)
    d_mse.append(delta_mse)
    n_train = len(train_sample_origin)
    print(n_train)
    n_train_points.append(n_train)

output_dict = {
    'train_losses': train_losses,
    'test_losses': test_losses, 
    'n_train_points': n_train_points,
    'mse_before': mse_before,
    'mse_after': mse_after,
    'd_mse': d_mse, 
    'd_uncert': d_train_uncert
}

output_path = os.path.join(save_paths['outputs'], f"pipeline_outputs_lam_{lam}.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(output_dict, f)
