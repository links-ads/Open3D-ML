import yaml
from argparse import ArgumentParser
import ast
import ruamel.yaml
from pathlib import Path

def update_yaml_field(file_path, field, value):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True

    with open(file_path, 'r') as file:
        data = yaml.load(file)

    # Naviga attraverso i livelli del campo se è annidato
    keys = field.split('.')
    d = data
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

    with open(file_path, 'w') as file:
        yaml.dump(data, file)

if __name__ == "__main__":
    parser = ArgumentParser(description="Aggiorna un campo specifico in un file YAML")
    parser.add_argument("--file_path", type=Path, help="Input file YAML")
    parser.add_argument("--field", type=str, help="Campo da aggiornare (può essere annidato, separato da punti)")
    parser.add_argument("--value", type=str, help="Nuovo valore per il campo (come stringa che rappresenta un dizionario)")

    args = parser.parse_args()

    file_path = args.file_path
    field = args.field
    value = ast.literal_eval(args.value)
   

    update_yaml_field(file_path, field, value)
 
   