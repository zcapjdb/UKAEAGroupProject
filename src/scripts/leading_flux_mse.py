import comet_ml
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

from scripts.QLKNN import QLKNN, QLKNN_Big, QLKNNDataset
from scripts.utils import train_keys, target_keys, prepare_model, callbacks
import copy


TRAIN_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
VAL_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"

SIZES = [
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    1_000_000,
    10_000_000,
]

PARAMS = {
    "epochs": 50,
    "learning_rate": 0.001,
}

accelerator = "gpu"
num_gpu = 3


def main():
    leading_fluxes = ["efeetg_gb", "efetem_gb", "efiitg_gb"]

    for flux in leading_fluxes:
        keys = train_keys + [flux]
        train_data, val_data = prepare_model(TRAIN_PATH, VAL_PATH, QLKNNDataset, keys)

        for size in SIZES:
            print(f"Training model for {flux} with {size} training points")
            experiment_name = f"{flux}-{size}"
            train_data_size = copy.deepcopy(train_data)
            train_data_size.data = train_data_size.data.sample(size)

            batch_size = size if size <= 10_000 else 4096
            patience = 25 if size <= 10_000 else 5

            model = QLKNN_Big(n_input=15, **PARAMS, batch_size=batch_size)

            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=10,
            )

            val_loader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=10,
            )

            progress = TQDMProgressBar(refresh_rate=250)

            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=0.0, patience=patience
            )

            log_dir = f"/share/rcifdata/jbarr/UKAEAGroupProject/logs/RMS"
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=log_dir,
                filename=f"{flux}-{size}-" + "{epoch}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
            )

            trainer = Trainer(
                max_epochs=PARAMS["epochs"],
                accelerator=accelerator,
                strategy=DDPPlugin(find_unused_parameters=False),
                devices=num_gpu,
                callbacks=[early_stop_callback, checkpoint_callback, progress],
                log_every_n_steps=250,
                benchmark=True,
                check_val_every_n_epoch=5,
            )

            trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
