# *V*ae *A*ssisted *L*igand *D*isc*O*very (Valdo)

[![PyPI](https://img.shields.io/pypi/v/rs-valdo?color=blue)](https://pypi.org/project/rs-valdo/)

Using variational Autoencoders to improve the signal-to-noise ratio of drug
fragment screens

- [*V*ae *A*ssisted *L*igand *D*isc*O*very (Valdo)](#vae-assisted-ligand-discovery-valdo)
  - [Installation](#installation)
  - [*VALDO* Usage](#valdo-usage)
    - [Step 1: Diffraction Data](#step-1-diffraction-data)
    - [Step 2: Reindexing \& Scaling](#step-2-reindexing--scaling)
    - [Step 3: Normalization](#step-3-normalization)
    - [Step 4: VAE Training](#step-4-vae-training)
    - [Steps 5 \& 6: Reconstruction of "Apo" Data \& Calculating Difference Maps](#steps-5--6-reconstruction-of-apo-data--calculating-difference-maps)
    - [Steps 7 \& 8: Gaussian Blurring \& Searching for Blobs](#steps-7--8-gaussian-blurring--searching-for-blobs)
    - [Step 9: Identifying Events](#step-9-identifying-events)


## Installation

1. Create a environment conda or mamba
    ```
    mamba create -n valdo python=3.10
    ```

2. Install [pytorch](https://pytorch.org/get-started/locally/)

3. Install the package
    ```
    pip install rs-valdo
    ```
    
If you want the codes for further developing, install by:

```
git clone https://github.com/Hekstra-Lab/drug-screening.git
cd drug-screening/
pip install -e .
```

## *VALDO* Usage

The full flow chart is shown below, followed by a discussion of each of the steps.

![VALDO Flow Chart](./valdo/flowchart.png)

<br/>

### Step 1: Diffraction Data

The first step involves acquiring diffraction datasets in the `mtz` format. These datasets should follow a specific naming convention, where each file is named with a number followed by the `.mtz` extension (e.g., `01.mtz`, `02.mtz`, etc.).

**Usage:**

1. Ensure that you have collected diffraction datasets in the `mtz` format.

2. Organize the datasets with sequential numerical names (e.g., `01.mtz`, `02.mtz`, etc.).

Following this naming convention will allow datasets to be ready for further processing.

<br/>

### Step 2: Reindexing & Scaling

This step focuses on reindexing and scaling a list of input MTZ files to a reference MTZ file using gemmi. 

**Reindexing:** The datasets provided may include samples from different space groups that describe the same physical crystal structure. To ensure comparability, we reindex each sample to a common indexing scheme by applying reindexing operators. 

**Scaling:** The samples are scaled to a reference dataset using a global anisotropic scaling factor by an analytical scaling method that determines the Debye-Waller Factor. The scaling process ensures that structure factor amplitudes are comparable across different datasets, accounting for variabilities such as differences in lattice orientations.


**Usage:**

1. Import the required library, `valdo`.

2. Call the `reindex_files()` function from `valdo.reindex`. The `reindex_files()` function will enumerate possible reindexing operations for any space group and apply them to each input MTZ file. It will select the operation with the highest correlation with the reference dataset. The reindexed files will be saved in the specified output folder, following the same `##.mtz` naming convention.

    This function can be called with the following parameters:
    - `input_files`: List of paths to input MTZ files to be reindexed.
    - `reference_file`: Path to the reference MTZ file.
    - `output_folder`: Path to the folder where the reindexed MTZ files will be saved.
    - `columns`: A list containing the names of the columns in the dataset that represent the amplitude and the error column.

3. Create a `Scaler` object by providing the path to the reference MTZ file.

4. Call the `batch_scaling()` method of the `Scaler` object. The `batch_scaling()` method will apply the scaling process to each input MTZ file and save the scaled MTZ files in the specified output folder. Scaling metrics, such as least squares values and correlations, will be saved in the report file.

    This function can be called with the following parameters:
    - `mtz_path_list`: List of paths to input MTZ files to be scaled.
    - `outputmtz_path`: Path to the folder where the scaled MTZ files will be saved (optional, default `./scaled_mtzs/`).
    - `reportfile`: Path to the file where scaling metrics will be saved (optional, default `./scaling_data.json`).
    - `verbose`: Whether to display verbose information during scaling (optional, default `True`).
    - `n_iter`: Number of iterations for the analytical scaling method (optional, default `5`).

<details>
<summary>Code Example:</summary>

```python
import valdo

file_list = [list of input MTZ file paths]

reindexed_path = "path/to/output/folder"
scaled_path = "path/to/output/folder"

amplitude_col = "name_of_column_with_amplitudes"
error_col = "name_of_column_with_errors"

valdo.reindex.reindex_files(input_files=file_list, 
                            reference_file=file_list[0], 
                            output_folder=reindexed_path,
                            columns=[amplitude_col, error_col])

file_list = [list of reindexed MTZ file paths]

scaler = valdo.Scaler(reference_mtz=file_list[0])
metrics = scaler.batch_scaling(mtz_path_list=file_list, 
                                outputmtz_path=scaled_path, 
                                verbose=False)
```
</details><br/>

### Step 3: Normalization

This step involves normalizing the scaled structure factor amplitudes obtained in the previous step. The input is restricted to only those Miller indices present in the intersection of all datasets, and the VAE predicts structure factor amplitudes for all Miller indices in the union of all datasets.

Additionally, we standardize all the input data, such that the structure factor amplitudes for each Miller index in the union of all datasets have a mean of zero and a unit variance across datasets. 

**Usage:**

1. Import the required library, `valdo.preprocessing`.

2. Find the intersection and union of the scaled datasets using the following functions:

   - `find_intersection()`: Finds the intersection of `amplitude_col` from multiple input MTZ files and saves the result to the specified output pickle file. Arguments include the following:

      - `input_files`: List of input MTZ file paths.
      - `output_path`: Path to save the output pickle file containing the intersection data.
      - `amplitude_col`: Name of the column in the dataset that represents the scaled amplitude (default 'F-obs-scaled').

   - `find_union()`: Finds the union of `amplitude_col` from multiple input MTZ files and saves the result to the specified output pickle file. Arguments are the same as `find_intersection()`.

3. Generate the VAE input and output data using the `generate_vae_io()` function. This standardizes the intersection dataset using mean and standard deviation calculated from the union dataset. The standardized intersection becomes the VAE input, while the standardized union becomes the VAE output. Both the VAE input and output are saved to the specified folder. 

    This function can be called with the following parameters:

    - `intersection_path`: Path to the intersection dataset pickle file.
    - `union_path`: Path to the union dataset pickle file.
    - `io_folder`: Path to the output folder where the VAE input and output will be saved. Mean and standard deviation data calculated from the union dataset will also be saved in this folder as `union_mean.pkl` and `union_sd.pkl`.

<details>
<summary>Code Example:</summary>

```python
import valdo.preprocessing

file_list = [list of input MTZ file paths]
amplitude_scaled_col = "name_of_column_with_scaled_amplitudes"
intersection_path = "path/to/intersection_data.pkl"
union_path = "path/to/union_data.pkl"
vae_folder = "path/to/vae_input_output_folder"

valdo.preprocessing.find_intersection(input_files=file_list, 
                                      output_path=intersection_path,
                                      amplitude_col=amplitude_scaled_col)

valdo.preprocessing.find_union(input_files=file_list, 
                               output_path=union_path,
                               amplitude_col=amplitude_scaled_col)

valdo.preprocessing.generate_vae_io(intersection_path=intersection_path, 
                                    union_path=union_path, 
                                    io_folder=vae_folder)
```
</details><br/>

### Step 4: VAE Training

In this step, we train the VAE model using the provided VAE class. 

**Usage:**

1. Load the VAE input and output data that was generated in the previous step.

2. Initialize the VAE model with the desired hyperparameters. Tune-able hyperparameters include the following:
    - `n_dim_latent`: Number of dimensions in the latent space (optional, default `1`).

    - `n_hidden_layers`: Number of hidden layers in the encoder and decoder. If an int is given, it will applied to both encoder and decoder; If a length 2 list is given, first int will be used for encoder, the second will be used for decoder.

    - `n_hidden_size`: Number of units in hidden layers. If an int is given, it will be applied to all hidden layers in both encoder and decoder; otherwise, an array with length equal to the number of hidden layers can be given, the number of units will be assigned accordingly.

    - `activation` : Activation function for the hidden layers (optional, default `tanh`).

3. Split the data into training and validation sets. Randomly select a subset of indices for training and use the rest for validation.

4. Convert the data into PyTorch tensors.

5. Set up the optimizer for training.

6. Train the VAE model using the `train()` method. The training process involves minimizing the ELBO (Evidence Lower Bound) loss function, which consists of a Negative Log-Likelihood (NLL) term and a Kullback-Leibler (KL) divergence term. Arguments used in this function include:

    - `x_train`: Input data for training the VAE, a PyTorch tensor representing the VAE input data. 

    - `y_train`: Output data for training the VAE, a PyTorch tensor representing the VAE output data. 

    - `optim`: The optimizer used for training the VAE, a PyTorch optimizer object, such as `torch.optim.Adam`, that specifies the optimization algorithm and its hyperparameters, including the learning rate (`lr`).

    - `x_val`: Input data for validation during training. (optional, default `None`).

    - `y_val`: Output data for validation during training. (optional, default `None`).

    - `epochs`: The number of training epochs (epoch: a single pass through the data).

    - `batch_size`: The batch size used during training. If an integer is provided, the same batch size will be used for all epochs. If a list of integers is provided, it should have the same length as the number of epochs, and each value in the list will be used as the batch size for the corresponding epoch (optional, default `256`).

    - `w_kl`: The weight of the Kullback-Leibler (KL) divergence term in the ELBO loss function. The KL divergence term encourages the latent distribution to be close to a prior distribution (usually a standard normal distribution). A higher value of `w_kl` will increase the regularization strength on the latent space (optional, default `1.0`).

    **Note:** The VAE class internally keeps track of the training loss (`loss_train`) and its components (NLL and KL divergence) during each batch of training. These values can be accessed after training to monitor the training progress and performance. The `loss_train` attribute of the VAE object will be a list containing the training loss values for each batch during training. The `loss_names` attribute contains the names of the loss components: "Loss", "NLL", and "KL_div". These attributes are updated during training and can be used for analysis or visualization.

7. Save the trained VAE model for future use (optional).

<details>
<summary>Code Example with Pre-selected Hyperparamters:</summary>

```python
vae_input = np.load('path/to/vae_input.npy')
vae_output = np.load('path/to/vae_output.npy')

vae = valdo.VAE(n_dim_i=vae_input.shape[1], 
                n_dim_o=vae_output.shape[1], 
                n_dim_latent=3, 
                n_hidden_layers=[3, 6], 
                n_hidden_size=100, 
                activation=torch.relu)

# Randomly select 1300 indices for training
choice = np.random.choice(vae_input.shape[0], 1300, replace=False)    
train_ind = np.zeros(vae_input.shape[0], dtype=bool)
train_ind[choice] = True
test_ind = ~train_ind

# Split the input and output data into training and validation sets
x_train, x_val = vae_input[train_ind], vae_input[test_ind]
y_train, y_val = vae_output[train_ind], vae_output[test_ind]

# Convert the data to torch tensors
x_train, x_val, y_train, y_val = torch.tensor(x_train), torch.tensor(x_val), torch.tensor(y_train), torch.tensor(y_val)

# Set up the optimizer and train the VAE
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
vae.train(x_train, y_train, optimizer, x_val, y_val, epochs=300, batch_size=100, w_kl=1.0)

# Save the trained VAE model
vae.save('path/to/trained_vae.pkl')
```
</details><br/>

### Steps 5 & 6: Reconstruction of "Apo" Data & Calculating Difference Maps

In this step, VAE outputs are re-scaled accordingly to recover the original scale, and differences in amplitudes between the original and reconstructed data are calculated. A `recons` and a `diff` column will be created for all datasets.

**Usage:**

To perform the reconstruction, or re-scaling, the `rescale()` function can be called, providing the necessary arguments:

- `recons_path`: Path to the reconstructed output of the VAE in NumPy format.
- `intersection_path`: Path to the pickle file containing the intersection of all scaled datasets.
- `union_path`: Path to the pickle file containing the union data of all scaled datasets.
- `input_files`: List of input file paths. This list should be in the same order as is in the `vae_input.npy` or `intersection.mtz`.
- `info_folder`: Path to the folder containing files with the mean and SD used for standardization previously.
- `output_folder`: Path to the folder where the reconstructed data will be saved.
- `amplitude_col`: Column in the MTZ file that contains structure factor amplitudes to calculate the difference column.

<details>
<summary>Code Example:</summary>

```python
recons = vae.reconstruct(tensor object of vae_input)
recons = recons.detach().cpu().numpy()
np.save("path/to/reconstructed_vae.npy", recons)

valdo.preprocessing.rescale(recons_path="path/to/reconstructed_vae.npy", 
                            intersection_path="path/to/intersection.pkl", 
                            union_path="path/to/union.pkl", 
                            input_files=["path/to/data/01.mtz", "path/to/data/02.mtz", ...], 
                            info_folder="path/to/info_folder", 
                            output_folder="path/to/output_folder",
                            amplitude_col="name_of_column_with_scaled_amplitudes")
```
</details><br/>

### Steps 7 & 8: Gaussian Blurring & Searching for Blobs

**Note Regarding Phases:** In this section, phases are required for each dataset. You can obtain phases by completing refinement via PHENIX for each dataset, and utilizing those phases.  

**Note Regarding Models:** In this section, models are also required for each dataset. These can also be obtained by refinement via PHENIX for each dataset, and they should be stored in a single folder, with the same naming convention (i.e. `##.mtz`).

We offer a command-line tool for automatic refinement using PHENIX. Based on our tests, starting with a single apo model yields satisfactory phases and models for the following real-space maps. You can find an example `refine_drug.eff` file in the `notebook/` directory.

<details>
<summary>Code Example:</summary>

```shell
valdo.refine --pdbpath "xxx/xxx_apo.pdb" --mtzpath "xxx/*.mtz" --output "yyy/" --eff "xxx/refine_drug.eff"
```
</details><br/>


In this step, we aim to identify significant changes in electron density caused by ligand binding to a protein. By taking the absolute value of the electron density difference maps and applying Gaussian blurring, a new map is created with merged positive electron density blobs. The blurring process attempts to reduce noise. Blobs are then identified and characterized above a specified contour level and volume threshold.

**Usage:**
To generate blobs from electron density maps, call the `generate_blobs()` function, which takes electron density map files and corresponding refined protein models as inputs. The function preprocesses the maps and identifies blobs above a specified contour level and volume threshold (the volume threshold is the default set by `gemmi`). The output is a DataFrame containing statistics for each identified blob, including peak value, score, centroid coordinates, volume, and radius. 

This function can be called with the following arguments:

- `input_files`: List of input file paths.
- `model_folder`: Path to the folder containing the refined models for each dataset (pdb format).
- `diff_col`: Name of the column representing diffraction values in the input MTZ files.
- `phase_col`: Name of the column representing phase values in the input MTZ files.
- `output_folder`: Path to the output folder where the blob statistics DataFrame will be saved.
- `cutoff`: Blob cutoff value. Blobs with values below this cutoff will be ignored (optional, default `5`).
- `negate`: Whether to negate the blob statistics (optional, default `False`). Use True if there is interest in both positive and negative peaks, which is not typically of interest here due to the absolute value function applied to the map.
- `sample_rate`: Sample rate for generating the grid in the FFT process (optional, default `3`).

<details>
<summary>Code Example (3.5 Selected as Desired Cutoff):</summary>

```python
valdo.blobs.generate_blobs(input_files=["path/to/data/01.mtz", "path/to/data/02.mtz", ...],
                           model_folder="path/to/refined_models_folder",
                           diff_col="difference_column_name",
                           phase_col="phase_column_name",
                           output_folder="path/to/output_folder",
                           cutoff=3.5)
```
</details><br/>

### Step 9: Identifying Events

In this final step, the highest scoring blobs returned in the previous step can be analyzed individually. If the blob is plausibly a ligand, refinement with a ligand may be completed to determine whether or not the blob can be considered a "hit." 

Blobs that are returned can be related to various other events, not just ligand binding. Examples may include ligand-induced conformational change (which would still indicate the presence of a ligand) or various other unrelated conformational changes, such as radiation damage or cysteine oxidation (as is seen in `pipeline.ipynb`).
