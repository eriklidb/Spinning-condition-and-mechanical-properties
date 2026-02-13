# ML Approaches for Predicting Mechanical Properties of Artificial Spider Silk from Spinning Conditions

## About
Materials concerning the work "ML Approaches for Predicting Mechanical Properties of Artificial Spider Silk from Spinning Conditions",
by Erik Lidbjörk, Hedvig Kjellström, and Neeru Dubey. Supplementary material referenced in the report can be found in "supplementary.pdf".
The Spidrospin dataset containing the data for the artificial spider silk experiments and fibers is (for now) confidential, and can unfortunately not be shared. The PLM vector embeddings however, are shared. Given the dataset, all the experiments can be reproduced by utilizing a virtual environment and 
the requirenments file(s). The artificial spider data will be shared in the future if possible.

## Structure
* The code for the data processing and model training/evaluation can be found in src. 
* The generated figures can be found in figures. 
* The trained outer-fold models can be found in models.

### Code structure
* The ``data_processing`` module cleans the raw data found in multiple spreadsheets, makes it suitable for training, and saves it to another format (csv/hdf/xlsx). 
* The ``compare_properties`` module searches for matches for the protein sequences in the dataset using Smith-Waterman local alignment algorithm.
* The ``protein_sequences`` module runs code to derive vector embeddings from the corresponding protein sequences in the dataset. Utilizes the ``torch``and ``transformers`` modules.
* The ``dataset module`` wraps the cleaned dataset into a class. 
* The ``model_trainer`` module sets up the training scheme and model strucutre. 
* The ``evalute_models`` conducts the main experiments, which compares the SpinML and SeqSpinML model across metrics, generates comparative plots, shows SHAP importance values across spinnig conditions, and conducts an ablation study. 
Any remaining module is legacy code.

## Installation guide
If you want to play around with the code yourself, you can follow these steps on your terminal:
1. Clone the repository: 
```
git clone https://github.com/eriklidb/Spinning-condition-and-mechanical-properties
```
3. Create a virtual environment: 
```
python -m venv spidro_env
```
4. Activate it. On Linux/MacOS: 
```
source spidro_env/bin/activate
``` 
On Windows 
```
spidro_env\Scripts\activate
```
5. Install core dependencies: 
```
python -m pip install --upgrade pip 
python -m pip install -r requirements.txt
```
6. (Optional) installing ``torch`` and ``transformers`` will enable you to run the ``protein_sequences`` module and reproduce the PLM-derived embeddings:
```
python -m pip install -r requirements-plm.txt
```
