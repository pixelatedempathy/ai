"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import ast
import sys
from ast import literal_eval


def _apply_config_file(config_file: str, target_globals: dict) -> None:
    with open(config_file, encoding="utf-8") as f:
        source = f.read()

    module = ast.parse(source, filename=config_file, mode="exec")
    for node in module.body:
        # allow module docstring
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue

        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            key = node.targets[0].id
            if key not in target_globals:
                raise ValueError(f"Unknown config key in file: {key}")
            attempt = ast.literal_eval(node.value)
            assert type(attempt) == type(target_globals[key])
            print(f"Overriding: {key} = {attempt}")
            target_globals[key] = attempt
            continue

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            key = node.target.id
            if key not in target_globals:
                raise ValueError(f"Unknown config key in file: {key}")
            if node.value is None:
                raise ValueError(f"Missing value for config key: {key}")
            attempt = ast.literal_eval(node.value)
            assert type(attempt) == type(target_globals[key])
            print(f"Overriding: {key} = {attempt}")
            target_globals[key] = attempt
            continue

        raise ValueError(
            f"Unsupported statement in config file {config_file}. "
            "Only simple assignments of Python literals are allowed."
        )


def apply_overrides(target_globals: dict, argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    for arg in argv:
        if "=" not in arg:
            # assume it's the name of a config file
            assert not arg.startswith("--")
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file, encoding="utf-8") as f:
                print(f.read())
            _apply_config_file(config_file, target_globals)
        else:
            # assume it's a --key=value argument
            assert arg.startswith("--")
            key, val = arg.split("=", 1)
            key = key[2:]
            if key in target_globals:
                try:
                    # attempt to parse it (e.g. if bool, number, list, etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(target_globals[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                target_globals[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
