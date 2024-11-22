import torch
import torch.nn as nn
import torch.nn.functional as F
from ....datasets.utils import DataProcessing
import numpy as np
from .points import Points as pt


def filter_valid_label(scores, labels, num_classes, ignored_label_inds, device,confidence=None):
    """Loss functions for semantic segmentation."""
    valid_scores = scores.reshape(-1, num_classes).to(device)
    valid_labels = labels.reshape(-1).to(device)
    valid_confidence = None
    if confidence is not None:
        valid_confidence= confidence.reshape(-1).to(device)

    ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
    for ign_label in ignored_label_inds:
        ignored_bool = torch.logical_or(ignored_bool, torch.eq(valid_labels, ign_label))
    valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)

    valid_scores = torch.gather(
        valid_scores, 0, valid_idx.unsqueeze(-1).expand(-1, num_classes)
    )
    valid_labels = torch.gather(valid_labels, 0, valid_idx)
    if confidence is not None and valid_confidence is not None:
        valid_confidence = torch.gather(valid_confidence, 0, valid_idx)

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, num_classes, dtype=torch.int64)
    inserted_value = torch.zeros([1], dtype=torch.int64)

    for ign_label in ignored_label_inds:
        if ign_label >= 0:

            reducing_list = torch.cat(
                [reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]],
                0,
            )
    valid_labels = torch.gather(reducing_list.to(device), 0, valid_labels.long())
    
    return valid_scores, valid_labels, valid_confidence


class SemSegLoss(object):
    """Loss functions for semantic segmentation."""

    def __init__(self, pipeline, model, dataset, device):
        super(SemSegLoss, self).__init__()
        # weighted_CrossEntropyLoss
        if (
            "class_weights" in dataset.cfg.keys()
            and dataset.cfg.class_weights is not None
            and len(dataset.cfg.class_weights) != 0
        ):
            class_wt = DataProcessing.get_class_weights(dataset.cfg.class_weights)
            self.weights = torch.tensor(class_wt, dtype=torch.float, device=device)
            self.weighted_CrossEntropyLoss = nn.CrossEntropyLoss(weight=self.weights)
        else:
            self.weighted_CrossEntropyLoss = nn.CrossEntropyLoss()
    
    def weighted_confidence_CrossEntropyLoss(self, logits, labels, confidence, all_confidence):
   
        unweighted_losses = F.cross_entropy(logits, labels, reduction='none')
      
        
        weighted_confidence_losses = unweighted_losses * confidence
        
       
        if self.weights is None:
            output_loss = torch.mean(weighted_confidence_losses)
        else:
            
            #TODO num classes
            self.num_classes = 5
            one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
            weights = torch.sum(self.weights * one_hot_labels, dim=1)
            weighted_losses = weighted_confidence_losses * weights
            output_loss = torch.mean(weighted_losses)

        return output_loss

 # for i in range(len(confidence)):
        #     label = labels[i].item()
        #     if label == 0:
        #         continue  
        #     if label in all_confidence:
        #         confidence_value = all_confidence[label]
        #         #TODO numero passato da parametro
        #         points=pt(confidence_value, 1, 0.2, 1)
        #         #puoi usare una funzione lineare o esponenziale
        #         w_confidence = points.calculate_function(confidence[i],linear = True)
        #         weighted_confidence[i] = w_confidence
        #     else:
        #         raise KeyError(f"Label {label} don't have a confidence value")
   
        # weighted_confidence_losses = unweighted_losses * weighted_confidence
        

class SoftmaxEntropyLoss(nn.Module):

    def __init__(self):
        super(SoftmaxEntropyLoss, self).__init__()

    def forward(self, y_pred):
        """Entropy of softmax distribution from logits."""
        return -(y_pred.softmax(2) * y_pred.log_softmax(2)).sum(2)


