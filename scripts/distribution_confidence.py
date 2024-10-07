from pathlib import Path
import laspy as lp
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from argparse import ArgumentParser
from matplotlib.ticker import FuncFormatter

def distribution_confidence(file: Path, output_dir: Path):
    
    data = lp.read(str(file))
    
    
    labels = data.classification
    confidence = data.confidence  
    
    
    unique_labels = np.unique(labels)
    
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')
    
    def y_formatter(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f} x 10e6'
        else:
            return f'{int(x)}'
    
    for label in unique_labels:
      
        label_mask = (labels == label)
        label_confidence = confidence[label_mask]
        

        mean_confidence = np.mean(label_confidence)
        
       
        plt.figure(figsize=(10, 6))
        sns.histplot(label_confidence, bins=30, kde=True, color='orange', label=f'Class {label}',stat='count')
        plt.axvline(mean_confidence, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'Distribution of Confidence for Class {label}', fontsize=15)
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Number of points', fontsize=12)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))
        plt.legend()
        
        
      
        output_file = output_dir / f'class_{label}_confidence.png'
        plt.savefig(output_file)
        plt.close()
    
   

parser = ArgumentParser()

parser.add_argument(
    "--input", type=Path, required=True, help="Input folder or file", dest="INPUT"
)
parser.add_argument("--output", type=Path, required=True, help="Output folder", dest="OUT")

args = parser.parse_args()

args.OUT.mkdir(parents=True, exist_ok=True)

if args.INPUT.is_dir():
    for file in args.INPUT.iterdir():
        distribution_confidence(file, args.OUT)
else:
    distribution_confidence(args.INPUT, args.OUT)
