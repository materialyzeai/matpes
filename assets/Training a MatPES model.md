# Introduction

This notebook demonstrates the fitting of a TensorNet FP using the MatPES 2025.2 PBE dataset. Fitting of other architectures in MatGL with either the PBE or r2SCAN datasets is similar.

**Important Note**: The training data sizes and maximum number of epochs chosen in the notebook are deliberately small so that the notebook will run within a reasonably short amount of time on a single CPU (~5-10 mins) for demonstration purposes. **The resulting model is not expected to be production quality**. When properly training a model, use the entire dataset with a much greater number of epochs.


```python
from __future__ import annotations

import collections
import json
import os
import shutil
from functools import partial

import lightning as pl
import numpy as np
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from dgl.data.utils import split_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
from matgl.models import TensorNet
from matgl.utils.training import PotentialLightningModule, xavier_init
from monty.io import zopen
from pymatgen.core import Structure
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from matpes.data import get_data
```


```python
data = get_data("PBE", download_atoms=True)
```

We need to load the atomic energies as the zero reference.


```python
with zopen("MatPES-PBE-atoms.json.gz", "rt") as f:
    isolated_energies_pbe = json.load(f)
isolated_energies_pbe = {d["elements"][0]: d["energy"] for d in isolated_energies_pbe}
len(isolated_energies_pbe)
```




    89




```python
# Initialize the lists for storing structures with energies, forces, and optional stresses
structures = []
labels = collections.defaultdict(list)

for d in tqdm(data):
    structures.append(Structure.from_dict(d["structure"]))
    labels["energies"].append(d["energy"])
    labels["forces"].append(d["forces"])
    labels["stresses"].append(
        (voigt_6_to_full_3x3_stress(np.array(d["stress"])) * -0.1).tolist()
    )
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 434712/434712 [00:54<00:00, 8021.74it/s]


# Loading the data into the the MGLDataSet


```python
# define the graph converter for periodic systems
element_types = DEFAULT_ELEMENTS
cry_graph = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = MGLDataset(structures=structures, converter=cry_graph, labels=labels)
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 434712/434712 [03:07<00:00, 2323.23it/s]


# Data split

For the purposes for demonstration, we are only going to use 2% of the data for trainining and 0.1% of the data for validation. In a real training, you should use a split such as 90%:10% or 95%:5%.


```python
training_set, validation_set, _ = split_dataset(
    dataset, [0.02, 0.001, 0.979], random_state=42, shuffle=True
)
# define the proper collate function for MGLDataLoader
collate_fn = partial(collate_fn_pes, include_line_graph=False, include_stress=True)
# initialize dataloader for training and validation
train_loader, val_loader = MGLDataLoader(
    train_data=training_set,
    val_data=validation_set,
    collate_fn=collate_fn,
    batch_size=32,
    num_workers=0,
)
```

# Model Setup

Here, we demonstrate the initialization of the TensorNet architecture. You may use any of the other architectures implemented in MatGL.


```python
model = TensorNet(
    element_types=element_types,
    is_intensive=False,
    rbf_type="SphericalBessel",
    use_smooth=True,
    units=128,
)
```


```python
# calculate scaling factor for training
train_graphs = []
energies = []
forces = []
for g, _lat, _attrs, lbs in training_set:
    train_graphs.append(g)
    energies.append(lbs["energies"])
    forces.append(lbs["forces"])
forces = torch.concatenate(forces)
rms_forces = torch.sqrt(torch.mean(torch.sum(forces**2, dim=1)))
# weight initialization
xavier_init(model)
# setup the optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True
)
scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
```

# Setup the potential lightning module

Note that the max_epochs is set to 2 here for demonstration purposes. In a real fitting, this number should be much larger (probably > 1000).


```python
# setup element_refs
energies_offsets = np.array(
    [isolated_energies_pbe[element] for element in DEFAULT_ELEMENTS]
)
# initialize the potential lightning module
lit_model = PotentialLightningModule(
    model=model,
    element_refs=energies_offsets,
    data_std=rms_forces,
    optimizer=optimizer,
    scheduler=scheduler,
    loss="l1_loss",
    stress_weight=0.1,
    include_line_graph=False,
)
# setup loggers
path = os.getcwd()
logger = CSVLogger(save_dir=path)
# setup checkpoints
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_Total_Loss",
    mode="min",
    filename="{epoch:04d}-best_model",
)
# setup trainer
trainer = pl.Trainer(
    logger=logger,
    callbacks=[
        EarlyStopping(monitor="val_Total_Loss", mode="min", patience=200),
        checkpoint_callback,
    ],
    max_epochs=2,
    accelerator="cpu",  # you can use gpu instead
    gradient_clip_val=2.0,
    accumulate_grad_batches=4,
    profiler="simple",
)
```

    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs


# Run the fit


```python
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```


      | Name  | Type              | Params | Mode
    ----------------------------------------------------
    0 | mae   | MeanAbsoluteError | 0      | train
    1 | rmse  | MeanSquaredError  | 0      | train
    2 | model | Potential         | 837 K  | train
    ----------------------------------------------------
    837 K     Trainable params
    0         Non-trainable params
    837 K     Total params
    3.352     Total estimated model params size (MB)
    69        Modules in train mode
    0         Modules in eval mode



    Sanity Checking: |                                                                                            …



    Training: |                                                                                                   …



    Validation: |                                                                                                 …



    Validation: |                                                                                                 …


    `Trainer.fit` stopped: `max_epochs=2` reached.
    FIT Profiler Report

    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    |  Action                                                                                                                                                                     	|  Mean duration (s)	|  Num calls      	|  Total time (s) 	|  Percentage %   	|
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    |  Total                                                                                                                                                                      	|  -              	|  18171          	|  611.61         	|  100 %          	|
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    |  run_training_epoch                                                                                                                                                         	|  296.55         	|  2              	|  593.11         	|  96.975         	|
    |  run_training_batch                                                                                                                                                         	|  1.0341         	|  544            	|  562.55         	|  91.979         	|
    |  [Strategy]SingleDeviceStrategy.backward                                                                                                                                    	|  0.61331        	|  544            	|  333.64         	|  54.551         	|
    |  [Strategy]SingleDeviceStrategy.training_step                                                                                                                               	|  0.41931        	|  544            	|  228.11         	|  37.296         	|
    |  [LightningModule]PotentialLightningModule.optimizer_step                                                                                                                   	|  1.0494         	|  136            	|  142.72         	|  23.335         	|
    |  [Strategy]SingleDeviceStrategy.validation_step                                                                                                                             	|  0.61214        	|  30             	|  18.364         	|  3.0026         	|
    |  [_TrainingEpochLoop].train_dataloader_next                                                                                                                                 	|  0.0097045      	|  544            	|  5.2792         	|  0.86317        	|
    |  [_EvaluationLoop].val_next                                                                                                                                                 	|  0.010562       	|  30             	|  0.31686        	|  0.051807       	|
    |  [Callback]TQDMProgressBar.on_train_batch_end                                                                                                                               	|  0.00048945     	|  544            	|  0.26626        	|  0.043535       	|
    |  [LightningModule]PotentialLightningModule.configure_gradient_clipping                                                                                                      	|  0.00061047     	|  136            	|  0.083024       	|  0.013575       	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_train_epoch_end       	|  0.037707       	|  2              	|  0.075415       	|  0.012331       	|
    |  [LightningModule]PotentialLightningModule.optimizer_zero_grad                                                                                                              	|  0.00039795     	|  136            	|  0.054121       	|  0.008849       	|
    |  [Strategy]SingleDeviceStrategy.batch_to_device                                                                                                                             	|  3.308e-05      	|  574            	|  0.018988       	|  0.0031046      	|
    |  [Callback]TQDMProgressBar.on_validation_batch_end                                                                                                                          	|  0.00043932     	|  30             	|  0.01318        	|  0.0021549      	|
    |  [LightningModule]PotentialLightningModule.transfer_batch_to_device                                                                                                         	|  2.1152e-05     	|  574            	|  0.012141       	|  0.0019851      	|
    |  [Callback]TQDMProgressBar.on_validation_start                                                                                                                              	|  0.0025428      	|  3              	|  0.0076283      	|  0.0012473      	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_train_batch_end       	|  1.1304e-05     	|  544            	|  0.0061496      	|  0.0010055      	|
    |  [Callback]TQDMProgressBar.on_train_start                                                                                                                                   	|  0.003221       	|  1              	|  0.003221       	|  0.00052664     	|
    |  [Callback]TQDMProgressBar.on_sanity_check_start                                                                                                                            	|  0.0032153      	|  1              	|  0.0032153      	|  0.00052572     	|
    |  [LightningModule]PotentialLightningModule.on_validation_model_zero_grad                                                                                                    	|  0.0015928      	|  2              	|  0.0031856      	|  0.00052086     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_train_epoch_end                                                                                     	|  0.0012048      	|  2              	|  0.0024095      	|  0.00039397     	|
    |  [Callback]TQDMProgressBar.on_validation_end                                                                                                                                	|  0.00079675     	|  3              	|  0.0023902      	|  0.00039081     	|
    |  [Callback]TQDMProgressBar.on_validation_batch_start                                                                                                                        	|  5.8543e-05     	|  30             	|  0.0017563      	|  0.00028716     	|
    |  [Callback]ModelSummary.on_fit_start                                                                                                                                        	|  0.001716       	|  1              	|  0.001716       	|  0.00028058     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_after_backward                                                                                      	|  2.3265e-06     	|  544            	|  0.0012656      	|  0.00020693     	|
    |  [Callback]TQDMProgressBar.on_train_epoch_start                                                                                                                             	|  0.00059594     	|  2              	|  0.0011919      	|  0.00019488     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_train_batch_end                                                                                     	|  1.2921e-06     	|  544            	|  0.00070288     	|  0.00011492     	|
    |  [Callback]TQDMProgressBar.on_train_epoch_end                                                                                                                               	|  0.00032792     	|  2              	|  0.00065583     	|  0.00010723     	|
    |  [Callback]ModelSummary.on_train_batch_end                                                                                                                                  	|  1.1395e-06     	|  544            	|  0.00061988     	|  0.00010135     	|
    |  [LightningModule]PotentialLightningModule.on_before_batch_transfer                                                                                                         	|  8.734e-07      	|  574            	|  0.00050133     	|  8.1969e-05     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_before_backward                                                                                     	|  9.0663e-07     	|  544            	|  0.00049321     	|  8.0641e-05     	|
    |  [LightningModule]PotentialLightningModule.on_validation_model_eval                                                                                                         	|  0.00013688     	|  3              	|  0.00041063     	|  6.7139e-05     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_train_batch_start                                                                                   	|  7.4417e-07     	|  544            	|  0.00040483     	|  6.6191e-05     	|
    |  [LightningModule]PotentialLightningModule.on_after_backward                                                                                                                	|  6.8329e-07     	|  544            	|  0.00037171     	|  6.0776e-05     	|
    |  [Callback]TQDMProgressBar.on_after_backward                                                                                                                                	|  6.4567e-07     	|  544            	|  0.00035124     	|  5.7429e-05     	|
    |  [LightningModule]PotentialLightningModule.on_train_batch_end                                                                                                               	|  5.6762e-07     	|  544            	|  0.00030878     	|  5.0487e-05     	|
    |  [LightningModule]PotentialLightningModule.on_train_batch_start                                                                                                             	|  5.3422e-07     	|  544            	|  0.00029061     	|  4.7516e-05     	|
    |  [Callback]TQDMProgressBar.on_train_batch_start                                                                                                                             	|  5.3293e-07     	|  544            	|  0.00028991     	|  4.7402e-05     	|
    |  [Callback]TQDMProgressBar.on_train_end                                                                                                                                     	|  0.00028108     	|  1              	|  0.00028108     	|  4.5958e-05     	|
    |  [Strategy]SingleDeviceStrategy.on_train_batch_start                                                                                                                        	|  5.0934e-07     	|  544            	|  0.00027708     	|  4.5304e-05     	|
    |  [LightningModule]PotentialLightningModule.on_before_backward                                                                                                               	|  5.0927e-07     	|  544            	|  0.00027704     	|  4.5297e-05     	|
    |  [Callback]ModelSummary.on_after_backward                                                                                                                                   	|  5.0422e-07     	|  544            	|  0.0002743      	|  4.4849e-05     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_after_backward        	|  4.7799e-07     	|  544            	|  0.00026003     	|  4.2515e-05     	|
    |  [Callback]ModelSummary.on_train_batch_start                                                                                                                                	|  4.6718e-07     	|  544            	|  0.00025414     	|  4.1553e-05     	|
    |  [Callback]TQDMProgressBar.on_before_backward                                                                                                                               	|  4.4253e-07     	|  544            	|  0.00024074     	|  3.9362e-05     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_train_batch_start     	|  4.355e-07      	|  544            	|  0.00023691     	|  3.8736e-05     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_before_backward       	|  4.2531e-07     	|  544            	|  0.00023137     	|  3.7829e-05     	|
    |  [Callback]ModelSummary.on_before_backward                                                                                                                                  	|  4.1274e-07     	|  544            	|  0.00022453     	|  3.6711e-05     	|
    |  [LightningModule]PotentialLightningModule.on_after_batch_transfer                                                                                                          	|  3.7404e-07     	|  574            	|  0.0002147      	|  3.5104e-05     	|
    |  [LightningModule]PotentialLightningModule.on_train_epoch_end                                                                                                               	|  9.475e-05      	|  2              	|  0.0001895      	|  3.0984e-05     	|
    |  [LightningModule]PotentialLightningModule.lr_scheduler_step                                                                                                                	|  8.5063e-05     	|  2              	|  0.00017013     	|  2.7816e-05     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_validation_end        	|  4.6291e-05     	|  3              	|  0.00013887     	|  2.2706e-05     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_before_zero_grad                                                                                    	|  9.5588e-07     	|  136            	|  0.00013        	|  2.1255e-05     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_before_optimizer_step                                                                               	|  7.6411e-07     	|  136            	|  0.00010392     	|  1.6991e-05     	|
    |  [LightningModule]PotentialLightningModule.on_before_zero_grad                                                                                                              	|  6.4674e-07     	|  136            	|  8.7957e-05     	|  1.4381e-05     	|
    |  [Callback]TQDMProgressBar.on_before_zero_grad                                                                                                                              	|  5.224e-07      	|  136            	|  7.1047e-05     	|  1.1616e-05     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_before_zero_grad      	|  5.0919e-07     	|  136            	|  6.925e-05      	|  1.1323e-05     	|
    |  [Callback]ModelSummary.on_before_zero_grad                                                                                                                                 	|  4.5806e-07     	|  136            	|  6.2296e-05     	|  1.0186e-05     	|
    |  [Callback]ModelSummary.on_before_optimizer_step                                                                                                                            	|  4.2549e-07     	|  136            	|  5.7867e-05     	|  9.4614e-06     	|
    |  [Callback]TQDMProgressBar.on_before_optimizer_step                                                                                                                         	|  4.0771e-07     	|  136            	|  5.5449e-05     	|  9.0661e-06     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_before_optimizer_step 	|  3.9732e-07     	|  136            	|  5.4035e-05     	|  8.835e-06      	|
    |  [LightningModule]PotentialLightningModule.on_before_optimizer_step                                                                                                         	|  3.4404e-07     	|  136            	|  4.679e-05      	|  7.6503e-06     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.setup                    	|  3.7875e-05     	|  1              	|  3.7875e-05     	|  6.1927e-06     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_validation_batch_end                                                                                	|  1.2403e-06     	|  30             	|  3.7208e-05     	|  6.0837e-06     	|
    |  [Callback]ModelSummary.on_validation_batch_end                                                                                                                             	|  9.2915e-07     	|  30             	|  2.7874e-05     	|  4.5576e-06     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_validation_batch_start                                                                              	|  7.8187e-07     	|  30             	|  2.3456e-05     	|  3.8352e-06     	|
    |  [LightningModule]PotentialLightningModule.on_validation_batch_end                                                                                                          	|  7.4999e-07     	|  30             	|  2.25e-05       	|  3.6788e-06     	|
    |  [Callback]ModelSummary.on_validation_batch_start                                                                                                                           	|  5.472e-07      	|  30             	|  1.6416e-05     	|  2.6841e-06     	|
    |  [LightningModule]PotentialLightningModule.on_validation_batch_start                                                                                                        	|  5.3756e-07     	|  30             	|  1.6127e-05     	|  2.6368e-06     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_validation_batch_end  	|  5.2648e-07     	|  30             	|  1.5795e-05     	|  2.5825e-06     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_validation_end                                                                                      	|  4.1527e-06     	|  3              	|  1.2458e-05     	|  2.0369e-06     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_validation_batch_start	|  4.1244e-07     	|  30             	|  1.2373e-05     	|  2.0231e-06     	|
    |  [Callback]TQDMProgressBar.on_sanity_check_end                                                                                                                              	|  1.1791e-05     	|  1              	|  1.1791e-05     	|  1.9279e-06     	|
    |  [LightningModule]PotentialLightningModule.on_fit_start                                                                                                                     	|  6.9581e-06     	|  1              	|  6.9581e-06     	|  1.1377e-06     	|
    |  [Callback]ModelSummary.on_validation_end                                                                                                                                   	|  1.5833e-06     	|  3              	|  4.75e-06       	|  7.7664e-07     	|
    |  [Callback]ModelSummary.on_validation_start                                                                                                                                 	|  1.1527e-06     	|  3              	|  3.458e-06      	|  5.654e-07      	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_validation_epoch_end                                                                                	|  1.125e-06      	|  3              	|  3.3751e-06     	|  5.5184e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_train_start                                                                                         	|  3.1251e-06     	|  1              	|  3.1251e-06     	|  5.1096e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.setup                                                                                                  	|  2.9171e-06     	|  1              	|  2.9171e-06     	|  4.7696e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_fit_start             	|  2.875e-06      	|  1              	|  2.875e-06      	|  4.7007e-07     	|
    |  [Callback]ModelSummary.on_train_epoch_start                                                                                                                                	|  1.2295e-06     	|  2              	|  2.4589e-06     	|  4.0204e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_train_end                                                                                           	|  2.3751e-06     	|  1              	|  2.3751e-06     	|  3.8834e-07     	|
    |  [Callback]TQDMProgressBar.setup                                                                                                                                            	|  2.292e-06      	|  1              	|  2.292e-06      	|  3.7475e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_validation_start                                                                                    	|  7.6399e-07     	|  3              	|  2.292e-06      	|  3.7475e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_save_checkpoint                                                                                     	|  1.1455e-06     	|  2              	|  2.2911e-06     	|  3.746e-07      	|
    |  [Callback]ModelSummary.on_train_epoch_end                                                                                                                                  	|  1.104e-06      	|  2              	|  2.2079e-06     	|  3.6101e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_train_epoch_start                                                                                   	|  1.0419e-06     	|  2              	|  2.0838e-06     	|  3.4071e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_validation_start      	|  6.6628e-07     	|  3              	|  1.9989e-06     	|  3.2682e-07     	|
    |  [Callback]ModelSummary.on_train_start                                                                                                                                      	|  1.959e-06      	|  1              	|  1.959e-06      	|  3.2031e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_fit_start                                                                                           	|  1.9171e-06     	|  1              	|  1.9171e-06     	|  3.1346e-07     	|
    |  [Callback]ModelSummary.on_sanity_check_start                                                                                                                               	|  1.834e-06      	|  1              	|  1.834e-06      	|  2.9987e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_sanity_check_start                                                                                  	|  1.8331e-06     	|  1              	|  1.8331e-06     	|  2.9971e-07     	|
    |  [LightningModule]PotentialLightningModule.on_validation_start                                                                                                              	|  5.834e-07      	|  3              	|  1.7502e-06     	|  2.8616e-07     	|
    |  [LightningModule]PotentialLightningModule.on_validation_end                                                                                                                	|  5.8332e-07     	|  3              	|  1.75e-06       	|  2.8612e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_fit_end                                                                                             	|  1.6249e-06     	|  1              	|  1.6249e-06     	|  2.6568e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_train_start           	|  1.5839e-06     	|  1              	|  1.5839e-06     	|  2.5898e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_validation_epoch_start                                                                              	|  5.2775e-07     	|  3              	|  1.5832e-06     	|  2.5887e-07     	|
    |  [Strategy]SingleDeviceStrategy.on_validation_end                                                                                                                           	|  5.2767e-07     	|  3              	|  1.583e-06      	|  2.5883e-07     	|
    |  [LightningModule]PotentialLightningModule.configure_optimizers                                                                                                             	|  1.542e-06      	|  1              	|  1.542e-06      	|  2.5213e-07     	|
    |  [LightningModule]PotentialLightningModule.on_validation_epoch_end                                                                                                          	|  5.1401e-07     	|  3              	|  1.542e-06      	|  2.5213e-07     	|
    |  [LightningModule]PotentialLightningModule.on_train_epoch_start                                                                                                             	|  7.4995e-07     	|  2              	|  1.4999e-06     	|  2.4524e-07     	|
    |  [Callback]ModelSummary.on_save_checkpoint                                                                                                                                  	|  6.8755e-07     	|  2              	|  1.3751e-06     	|  2.2483e-07     	|
    |  [Callback]TQDMProgressBar.on_validation_epoch_end                                                                                                                          	|  4.3105e-07     	|  3              	|  1.2931e-06     	|  2.1143e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_validation_epoch_end  	|  4.3066e-07     	|  3              	|  1.292e-06      	|  2.1124e-07     	|
    |  [Callback]ModelSummary.on_validation_epoch_end                                                                                                                             	|  4.1677e-07     	|  3              	|  1.2503e-06     	|  2.0443e-07     	|
    |  [Callback]ModelSummary.on_validation_epoch_start                                                                                                                           	|  4.0272e-07     	|  3              	|  1.2082e-06     	|  1.9754e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.on_sanity_check_end                                                                                    	|  1.2082e-06     	|  1              	|  1.2082e-06     	|  1.9754e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_validation_epoch_start	|  3.8906e-07     	|  3              	|  1.1672e-06     	|  1.9084e-07     	|
    |  [Callback]TQDMProgressBar.on_validation_epoch_start                                                                                                                        	|  3.8898e-07     	|  3              	|  1.1669e-06     	|  1.908e-07      	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_save_checkpoint       	|  5.8347e-07     	|  2              	|  1.1669e-06     	|  1.908e-07      	|
    |  [LightningModule]PotentialLightningModule.configure_callbacks                                                                                                              	|  1.1248e-06     	|  1              	|  1.1248e-06     	|  1.8391e-07     	|
    |  [LightningModule]PotentialLightningModule.on_validation_epoch_start                                                                                                        	|  3.6097e-07     	|  3              	|  1.0829e-06     	|  1.7706e-07     	|
    |  [Strategy]SingleDeviceStrategy.on_validation_start                                                                                                                         	|  3.4731e-07     	|  3              	|  1.0419e-06     	|  1.7036e-07     	|
    |  [Callback]TQDMProgressBar.on_save_checkpoint                                                                                                                               	|  5.2061e-07     	|  2              	|  1.0412e-06     	|  1.7024e-07     	|
    |  [Callback]EarlyStopping{'monitor': 'val_Total_Loss', 'mode': 'min'}.teardown                                                                                               	|  1e-06          	|  1              	|  1e-06          	|  1.635e-07      	|
    |  [Callback]TQDMProgressBar.on_fit_start                                                                                                                                     	|  9.581e-07      	|  1              	|  9.581e-07      	|  1.5665e-07     	|
    |  [LightningModule]PotentialLightningModule.setup                                                                                                                            	|  9.1689e-07     	|  1              	|  9.1689e-07     	|  1.4991e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_train_epoch_start     	|  4.3761e-07     	|  2              	|  8.7521e-07     	|  1.431e-07      	|
    |  [Callback]ModelSummary.setup                                                                                                                                               	|  8.7498e-07     	|  1              	|  8.7498e-07     	|  1.4306e-07     	|
    |  [Callback]ModelSummary.on_train_end                                                                                                                                        	|  8.7498e-07     	|  1              	|  8.7498e-07     	|  1.4306e-07     	|
    |  [LightningModule]PotentialLightningModule.on_save_checkpoint                                                                                                               	|  3.7544e-07     	|  2              	|  7.5088e-07     	|  1.2277e-07     	|
    |  [Strategy]SingleDeviceStrategy.on_train_start                                                                                                                              	|  7.4995e-07     	|  1              	|  7.4995e-07     	|  1.2262e-07     	|
    |  [Callback]TQDMProgressBar.teardown                                                                                                                                         	|  6.6613e-07     	|  1              	|  6.6613e-07     	|  1.0891e-07     	|
    |  [Callback]ModelSummary.on_sanity_check_end                                                                                                                                 	|  6.2515e-07     	|  1              	|  6.2515e-07     	|  1.0221e-07     	|
    |  [LightningModule]PotentialLightningModule.teardown                                                                                                                         	|  6.2515e-07     	|  1              	|  6.2515e-07     	|  1.0221e-07     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_sanity_check_start    	|  5.8301e-07     	|  1              	|  5.8301e-07     	|  9.5324e-08     	|
    |  [LightningModule]PotentialLightningModule.on_train_end                                                                                                                     	|  5.8301e-07     	|  1              	|  5.8301e-07     	|  9.5324e-08     	|
    |  [LightningModule]PotentialLightningModule.on_fit_end                                                                                                                       	|  5.8301e-07     	|  1              	|  5.8301e-07     	|  9.5324e-08     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_train_end             	|  5.4203e-07     	|  1              	|  5.4203e-07     	|  8.8624e-08     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_fit_end               	|  5.4203e-07     	|  1              	|  5.4203e-07     	|  8.8624e-08     	|
    |  [Callback]TQDMProgressBar.on_fit_end                                                                                                                                       	|  5.0012e-07     	|  1              	|  5.0012e-07     	|  8.1771e-08     	|
    |  [Callback]ModelSummary.teardown                                                                                                                                            	|  5.0012e-07     	|  1              	|  5.0012e-07     	|  8.1771e-08     	|
    |  [LightningModule]PotentialLightningModule.on_train_start                                                                                                                   	|  4.9989e-07     	|  1              	|  4.9989e-07     	|  8.1733e-08     	|
    |  [Strategy]SingleDeviceStrategy.on_train_end                                                                                                                                	|  4.5891e-07     	|  1              	|  4.5891e-07     	|  7.5033e-08     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.on_sanity_check_end      	|  4.17e-07       	|  1              	|  4.17e-07       	|  6.8181e-08     	|
    |  [Callback]ModelCheckpoint{'monitor': 'val_Total_Loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}.teardown                 	|  4.17e-07       	|  1              	|  4.17e-07       	|  6.8181e-08     	|
    |  [Callback]ModelSummary.on_fit_end                                                                                                                                          	|  3.7509e-07     	|  1              	|  3.7509e-07     	|  6.1329e-08     	|
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




```python
# save trained model
model_export_path = "./trained_model/"
lit_model.model.save(model_export_path)
```

# Cleanup


```python
# This code just performs cleanup for this notebook.
shutil.rmtree("MGLDataset")
shutil.rmtree("trained_model")
```
