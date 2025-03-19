from pathlib import Path
import logging
import laspy as lp
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from argparse import ArgumentParser
from matplotlib.ticker import FuncFormatter

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#single file
def distribution_confidence(file: Path, output_dir: Path):
    class_names = {
    1: 'Soil',
    2: 'Terrain',
    3: 'Vegetation',
    4: 'Building',
    5: 'Street Element',
    6: 'Water',
    }
    data = lp.read(str(file))
    
    
    labels = data.classification
    confidence = data.confidence  
    
    
    unique_labels = np.unique(labels)
    
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')
    
    def y_formatter(x, pos):
        return f'$10^{{{int(np.log10(x))}}}$' if x != 0 else '0'
    
    weighted_means = {}
    
    colors = ['#03A678', 'gray', '#064004', '#8C4126', '#814BA6', '#2E4BF2']
    
    captions = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Crea una griglia 2x3
    axes = axes.flatten() 
    for i, label in enumerate(unique_labels):
        label_mask = (labels == label)
        label_confidence = confidence[label_mask]
        
        
        unique_confidences, counts = np.unique(label_confidence, return_counts=True)
        weighted_mean_confidence = np.average(unique_confidences, weights=counts)
        
        weighted_mean_confidence = np.round(weighted_mean_confidence, 2)
        weighted_means[label] = weighted_mean_confidence
        
        ax = axes[i]
        sns.histplot(label_confidence, bins=20, kde=True, color=colors[i], label=class_names[label], stat='count', ax=ax,alpha=1.0)
        
        # ax.set_title(f'Distribution of Confidence for {class_names[label]}', fontsize=15)
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Number of points', fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
       

        ax.text(0.5, -0.15, captions[i], transform=ax.transAxes, fontsize=14, ha='center', va='top')

        ax.axvline(weighted_mean_confidence, color='black', linestyle='--', linewidth=2, label='Weighted Mean Confidence')
        ax.text(weighted_mean_confidence + 0.02, ax.get_ylim()[1] * 0.9, f'{weighted_mean_confidence:.2f}', 
                color='black', ha='left', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    
        ax.legend(fontsize='small')
        ax.grid(False)
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
     
    plt.tight_layout()
    output_file = output_dir / 'all_classes_confidence.png'
    plt.savefig(output_file)
    plt.close()
        
        
    return weighted_means


parser = ArgumentParser()

parser.add_argument(
    "--input", type=Path, required=True, help="Input folder or file", dest="INPUT"
)
parser.add_argument("--output", type=Path, required=True, help="Output folder", dest="OUT")

args = parser.parse_args()

args.OUT.mkdir(parents=True, exist_ok=True)

all_weighted_means = []
if args.INPUT.is_dir():
    for file in args.INPUT.iterdir():
        weighted_means=distribution_confidence(file, args.OUT)
        all_weighted_means.append(weighted_means)
else:
    weighted_means = distribution_confidence(args.INPUT, args.OUT)
    all_weighted_means.append(weighted_means)
   
total_means = {}
for wm in all_weighted_means:
    for label, mean in wm.items():
        if label not in total_means:
            total_means[label] = []
        total_means[label].append(mean)

average_means = {}
for label, means in total_means.items():
    average_means[label] = round(np.mean(means), 2)

output_log_file=args.OUT / 'average_confidence.txt'
with open(output_log_file, 'w') as f:
   
    for label, mean in average_means.items():
        if label == list(average_means.keys())[0]:
            f.write('{')
        f.write(f'{label}:{mean}')
        if label != list(average_means.keys())[-1]:
            f.write(',')
    f.write('}')

logger.info(f'Average weighted means: {average_means}')
logger.info(f'Summary written to {output_log_file}')


