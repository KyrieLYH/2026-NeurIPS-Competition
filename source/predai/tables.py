"""Resolve active release tables, without scanning raw or reproduction outputs."""
from pathlib import Path


def table_path(root, bench, table):
    folder = Path(root) / bench
    names = ('responses', 'response') if table in ('response', 'responses') else (table,)
    for directory in (folder / 'formatted_tables', folder):
        for name in names:
            path = directory / f'{name}.parquet'
            if path.is_file():
                return path
    raise FileNotFoundError(f'{bench}: missing {table} release table')


def discover(root):
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            for table in ('subjects', 'items', 'responses'):
                table_path(root, folder.name, table)
        except FileNotFoundError:
            continue
        found.append(folder.name)
    return found
