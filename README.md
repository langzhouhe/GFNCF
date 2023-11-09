# GFNCF
I. Environment: TensorFlow 1.14.0, Keras 2.2.5, Python 3.6
Data Preparation:
Configure the config.py file, mainly setting the dataset_name, rating_path, and sep. Note whether the original data file ends with .csv or .dat, and what the separator in the original data is.
Split the data with crossData.py.
Global Feature Extraction: DAE.py
Model Execution: model.py. Note when running the ml-10m dataset, set batch_size to 4096, for other datasets set to 1024.
Different datasets require setting different regularization parameters, recommended settings are as follows:
ml-1a: Set the Embedding layer with embeddings_regularizer=l2(0.003); for the #DMF part, set kernel_regularizer=l2(0.001); and for the MLP part, set kernel_regularizer=l2(0.005).
ml-1m: Set the Embedding layer with embeddings_regularizer=l2(0.00); for the #DMF part, set kernel_regularizer=l2(0.00); and for the MLP part, set kernel_regularizer=l2(0.004).
ml-10m: Set the Embedding layer with embeddings_regularizer=l2(0.00); for the #DMF part, set kernel_regularizer=l2(0.002); and for the MLP part, set kernel_regularizer=l2(0.01).
filmtrust: Set the Embedding layer with embeddings_regularizer=l2(0.003); for the #DMF part, set kernel_regularizer=l2(0.002); and for the MLP part, set kernel_regularizer=l2(0.005).
ciaodvd: Set the Embedding layer with embeddings_regularizer=l2(0.006); for the #DMF part, set kernel_regularizer=l2(0.01); and for the MLP part, set kernel_regularizer=l2(0.01).
If re-running the data split, the regularization parameters may need fine-tuning from this basis.

II. Quick Run:
Use the split dataset and global features, and directly run model.py.
