from contextlib import contextmanager
import pandas as pd


@contextmanager
def concat_csv(sources, destination, to_be_kept, to_rename={}, delimiter=',', sort_key=None):
    """
    Concatenate csv files in a unique file.

    Args:
    ----
        :sources: list-like
            Source filenames
        :destination: str
            Destination filename
        :to_be_kept: list
            Columns that have to be kept
        :to_rename: dictionnary
            Mapping between old column names and new ones
        :delimiter: str (default: ',')
        :sort_key: str (default: None)
            Column on which the resulting file has to be sorted (final name)

    """
    container = []

    for f in sources:
        _df = pd.read_csv(f, index_col=None, header=0, usecols=to_be_kept, delimiter=delimiter)
        container.append(_df)

    df = pd.concat(container, sort=False)

    if to_rename:   # Renaming
        df.rename(columns=to_rename, inplace=True)

    if sort_key:  # Sorting
        df.sort_values(sort_key)

    df.to_csv(destination, index=False)
    print('Files concatenated in {0} ({1} entries)'.format(destination, df.shape[0]))


@contextmanager
def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter)
