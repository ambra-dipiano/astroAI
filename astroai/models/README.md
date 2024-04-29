# Model training

After you have compiled the configuration file following the commentary in [template_cnn.yml](../conf/template_cnn.yml) an preprocess your data following the [README](../tools/REDME.md) you can train your models:

```bash
python cnn.py -f ../conf/your_conf_file.yml
```

Alternatively you can submit a job to [Slurm](https://slurm.schedmd.com/documentation.html) if available to you:

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml  -a cnn -m clean
```

or 

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -a cnn -m localise
```

