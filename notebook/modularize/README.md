# Modularize from Notebook Work

> This contains a set of Python and bash scripts that originated from experimenting in the notebook.
>
> The idea is that these scripts can be executed separately as needed, giving us more flexibility.

## Using the `.py` Files

> The `.py` files are meant to do a specific thing in the machine learning process.
> Most will read in and/or write out data from files. This could be expanded to be
> more flexible but it's fine for now since this is more of a demonstration.

## Running Scripts

> There are a couple of convenience bash scripts that will call the relevant Python
> scripts (`.py`)

You may need to make these executable with something like this command:
```sh
chmod +x *.sh
```

### Get clean data

> This will download the data, clean the data, and split into train & testing data.
> Intermediate data files are kept.

```sh
./get_clean_data.sh
```

### Get clean data

> This will train the model and evaluate the model using the relevant data from the
> `get_clean_data.sh` script.
> Intermediate data files are kept.


```sh
./train_and_evaluate.sh
```
