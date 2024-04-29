# Data preprocessing

After you have compiled the configuration file following the commentary in [template_cnn.yml](../conf/template_cnn.yml) you can execute the data preprocessing pipeline as follows:

```bash
python preprocess_ds.py -f ../conf/your_conf_file.yml
```

To pre-clean the processed dataset using a trained cnn-cleaner you can use the following:

```bash
python preclean_ds.py -f ../conf/your_conf_file.yml
```

Alternatively you can submit a job to [Slurm](https://slurm.schedmd.com/documentation.html) if available to you:

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -s preprocess_ds
```

or 

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -s preclean_ds
```

