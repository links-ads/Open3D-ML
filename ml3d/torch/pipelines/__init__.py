"""3D ML pipelines for torch."""

from .semantic_segmentation import SemanticSegmentation
from .object_detection import ObjectDetection
from .feature_extraction import FeatureExtraction

__all__ = ["SemanticSegmentation", "ObjectDetection", "FeatureExtraction"]
