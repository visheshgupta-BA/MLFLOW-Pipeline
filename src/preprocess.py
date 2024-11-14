import pandas as pd
import sys
import yaml
import os


params = yaml.safe_load(open('params.yaml'))['preprocess']

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f'Data preprocessing completed successfully. Output saved to {output_path}')



if __name__ == "__main__":
    preprocess_data(params['input'], params['output'])
