## Conda Env

Create and start the conda environment:

```sh
    conda env create -f environment.yml 
    conda activate xai-venv
```

Export env:
```sh
    conda env export | grep -v "^prefix: " > environment.yml
```
