# Reference pipeline

After you have compiled the configuration file following the commentary in [template_gp.yml](../conf/template_gp.yml) you can run the reference real-time pipeline for comparison:

```bash
python run_gammapy.py -f ../conf/your_conf_file.yml
```

Alternatively you can submit a job to [Slurm](https://slurm.schedmd.com/documentation.html) if available to you:

```bash
python slurmjobs.py -f ../conf/your_conf_file.yml -p gammapy
```

