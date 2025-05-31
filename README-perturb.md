# README

## Project Overview

This code applies various structured perturbations to Python code samples, generating diverse variants for inference with attacked models and for training and inference of classifiers.

## Features

* Supports a variety of perturbations including `false_if_fixed`, `false_if_random`, `false_loop_for`, `false_loop_while`, `unused_var_existing`, `unused_var_random`, `rename_var`, `rename_method`, `print_enter`, `print_variable`, and more
* Can traverse all perturbations or apply them randomly/sequentially by group (`false_if`, `false_loop`, `unused_var`, `rename`, `print`)
* Uses regex parsing of variable and method names to ensure perturbations do not alter original semantics
* Processes input files containing multi-line JSON samples and outputs a new JSONL file

## Quick Start

Place your original JSONLines file `input.jsonl` in the project directory. Each line should be formatted like:

```json
{
  "id": "0",
  "input": "<s> def foo(x):\\n    return x + 1",
  "gt": "def foo(x): return x + 1"
}
```

Run:

```bash
python perturb.py \
  --input_dir ./data/ \
  --input_file input.jsonl \
  --output_file perturbed.jsonl
```

After processing, `data/perturbed.jsonl` will contain the original samples plus multiple variants for each perturbation type.

## Command-Line Arguments

| Argument        | Type | Req. | Default | Description                 |
| --------------- | ---- | ---- | ------- | --------------------------- |
| `--input_dir`   | str  | Yes  | —       | Directory of the input file |
| `--input_file`  | str  | Yes  | —       | Input JSONL filename        |
| `--output_file` | str  | Yes  | —       | Output JSONL filename       |

## Perturbation Types

### Inserting Dead Control Branches (IDC)

* `false_if_fixed`: Insert a fixed `if False:` or `if "key"!="key":` statement
* `false_if_random`: Use the same `if` condition but randomly select the `if` body from existing code lines

### Inserting Redundant Variable Declarations (IRV)

* `unused_var_existing`: Declare a new variable initialized with an existing variable or constant
* `unused_var_random`: Declare a new variable initialized with a random constant

### Variable and Method Renaming (VR)

* `rename_var`: Randomly rename a variable to `var_<rand>`
* `rename_method`: Randomly rename a method to `method_<rand>`

### Inserting Debug Print Statements (IDP)

* `print_enter`: Insert `print("Debug: Entering method ...")` at the start of a method body
* `print_variable`: Insert `print("Debug: Variable ...")` immediately after the first occurrence of a variable definition

### Inserting Dead Loops (IDL)

* `false_loop_for`: Insert a `for` loop that never executes
* `false_loop_while`: Insert a `while` loop that never executes

## Example

```bash
python perturb.py --input_dir ./data/ --input_file input.jsonl --output_file perturbed.jsonl
# By default, generates for each sample: the original + all 12 variants
```
