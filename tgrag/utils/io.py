import os
import subprocess
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and capture its stdout and stderr as text.

    Parameters:
        cmd : list of str
            Command and arguments to execute.
        check : bool, optional
            If True, raise RuntimeError when the command fails (default: True).

    Returns:
        subprocess.CompletedProcess[str]
            The completed process object containing stdout, stderr, and return code.

    Raises:
        RuntimeError
            If `check` is True and the command exits with a non-zero status.
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p


def run_ext_sort(
    in_path: str | Path,
    out_path: str | Path,
    *,
    sort_cmd: str = 'sort',
    mem: str = '60%',
    tmpdir: str | Path,
    delimiter: Optional[str] = None,
    key_start_col: Optional[int] = None,
    key_numeric: bool = False,
    unique: bool = False,
) -> None:
    """Sort a text file using the external Unix `sort` command and write the result
    to an output file.

    Parameters:
        in_path : str or pathlib.Path
            Path to input vertex file.
        out_path : str or pathlib.Path
            Path where the sorted output will be written.
        sort_cmd : str, optional
            Name or path of the `sort` executable to run (default: ``"sort"``).
        mem : str, optional
            Memory limit passed to `sort` via ``-S`` (e.g. ``"60%"``).
        tmpdir : str or pathlib.Path
            Directory used by `sort` for temporary files.
        delimiter : str, optional
            Field delimiter to use for sorting. If ``None``, the comma delimiter is used.
        key_start_col : int, optional
            1-based index of the field to sort on. If ``None``, the entire line is
            used as the sort key.
        key_numeric : bool, optional
            If ``True``, perform a numeric sort on the selected key.
        unique : bool, optional
            If ``True``, suppress duplicate lines in the output (adds ``-u``).

    Returns: None

    Raises:
        RuntimeError
            If the `sort` command exits with a non-zero status. The error output from
            `sort` is included in the exception message.
    """
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    cmd = [sort_cmd, '-S', mem, '-T', str(tmpdir)]
    if delimiter is None:
        delimiter = ','
    cmd += ['-t', delimiter]
    if key_start_col is not None:
        k = f'{key_start_col},{key_start_col}' + ('n' if key_numeric else '')
        cmd += ['-k', k]
    if unique:
        cmd += ['-u']
    cmd += [str(in_path)]
    out_path = Path(out_path)
    with out_path.open('w', encoding='utf-8', newline='') as fout:
        p = subprocess.Popen(
            cmd, stdout=fout, stderr=subprocess.PIPE, text=True, env=env
        )
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"sort failed: {' '.join(cmd)}\n{err}")
