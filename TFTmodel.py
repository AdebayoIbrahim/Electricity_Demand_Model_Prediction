import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import seed_everything
from lightning.pytorch import Trainer
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE # Import available metrics

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# Load and clean
df = pd.read_csv('./caiso-electricity.csv')  # Update this
df.columns = df.columns.str.strip()
df['timestamp'] = pd.to_datetime(df['UTC Timestamp (Interval Ending)'])

# Only use key regions
regions = ['NP-15 LMP', 'SP-15 LMP', 'ZP-26 LMP']
df = df[['timestamp'] + regions]

# Reshape to long format
df_long = df.melt(id_vars=['timestamp'], value_vars=regions,
                  var_name='region', value_name='lmp')
df_long['time_idx'] = (df_long['timestamp'] - df_long['timestamp'].min()).dt.total_seconds() // 3600
df_long['time_idx'] = df_long['time_idx'].astype(int)

# Normalize
scaler = MinMaxScaler()
df_long['lmp'] = scaler.fit_transform(df_long[['lmp']])

# %% [perform prediction training]
max_encoder_length = 24  # past 24 hours
max_prediction_length = 1  # predict next hour

training_cutoff = df_long["time_idx"].max() - max_prediction_length

# Define dataset
tft_dataset = TimeSeriesDataSet(
    df_long[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="lmp",
    group_ids=["region"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["lmp"],
    time_varying_known_reals=["time_idx"],
    target_normalizer=GroupNormalizer(groups=["region"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Validation dataset
val_dataset = TimeSeriesDataSet.from_dataset(tft_dataset, df_long, predict=True, stop_randomization=True)
train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

# %% [run epoch trainings]
from torchmetrics import MeanSquaredError
 # Import pytorch_lightning

# Optional: reproducibility
seed_everything(42)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
lr_logger = LearningRateMonitor()

# Model
tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    learning_rate=0.001,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    loss=MeanSquaredError(),
    output_size=1,
    reduce_on_plateau_patience=2,
)

# Trainer
trainer = Trainer(
    max_epochs=20,
    accelerator="auto",
    gradient_clip_val=0.1,
    log_every_n_steps=1,
    # callbacks=[EarlyStopping ,lr_logger]
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
# %% [] errors]
# Predict on validation
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader).squeeze()

# Inverse transform (move tensors to CPU before NumPy conversion)
actuals_np = scaler.inverse_transform(actuals.cpu().numpy().reshape(-1, 1)).flatten()
preds_np = scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()



mae = mean_absolute_error(actuals_np, preds_np)
mse = mean_squared_error(actuals_np, preds_np)
rmse = np.sqrt(mse)
r2 = r2_score(actuals_np, preds_np)

print("📊 TFT Model Performance")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")

# %% [actual views of the model]
import pandas as pd
import matplotlib.pyplot as plt
import torch

# --- Step 1: Run prediction once and get raw outputs
# --- Step 1: Run prediction and get raw outputs
raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True)

# Extract prediction tensor
preds_tensor = raw_predictions.output.prediction  # [batch_size, prediction_length, 1]
x = raw_predictions.x

# Step 2: First predicted step
preds = preds_tensor[:, 0, 0].detach().cpu().numpy().flatten()

# Step 3: Actuals (aligned from x['decoder_target'])
actuals_tensor = x['decoder_target']  # [batch_size, prediction_length]
actuals = actuals_tensor[:, 0].detach().cpu().numpy().flatten()

# Step 4: Region (group_ids) from x["groups"]
region_ids = x["groups"][0].detach().cpu().numpy().flatten()

# Step 5: Map region IDs to names
region_encoder = tft_dataset.get_parameters()["categorical_encoders"]["__group_id__region"]
region_classes = list(region_encoder.classes_)
region_names = [region_classes[int(rid)] for rid in region_ids]

# Step 6: Inverse transform
actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

# Step 7: Plot
plot_df = pd.DataFrame({
    "Region": region_names,
    "Actual": actuals_inv,
    "Predicted": preds_inv
})

plt.figure(figsize=(18, 5))
for i, region in enumerate(plot_df["Region"].unique()):
    region_data = plot_df[plot_df["Region"] == region].reset_index(drop=True)
    if len(region_data) == 0:
        continue
    plt.subplot(1, 3, i + 1)
    plt.plot(region_data["Actual"], label="Actual", color='blue')
    plt.plot(region_data["Predicted"], label="Predicted", color='orange', linestyle='--')
    plt.title(f"Region: {region}")
    plt.xlabel("Sample Index")
    plt.ylabel("LMP")
    plt.legend()

plt.suptitle("TFT Model — Actual vs Predicted LMP by Region", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
