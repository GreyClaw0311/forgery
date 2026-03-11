#!/usr/bin/env python3
"""
Image Tamper Detection - Main Entry Point

Usage:
    python -m src.main --data tamper_data/easy --output results
    python -m src.main --image test.jpg --output results
    python -m src.main --data tamper_data/easy --detectors ELA DCT NOISE
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    ALL_DETECTORS,
    DEFAULT_DETECTORS,
    get_detector_by_name,
)
from src.utils import (
    load_image,
    save_image,
    load_mask,
    compute_iou,
    compute_precision_recall,
    compute_f1,
    threshold_feature_map,
    visualize_result,
    get_dataset_files,
)


def run_detector(detector, image: np.ndarray) -> np.ndarray:
    """Run a single detector on an image."""
    return detector.detect(image)


def evaluate_detector(
    feature_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold_method: str = 'otsu'
) -> Dict[str, float]:
    """Evaluate detector performance against ground truth."""
    # Threshold feature map
    pred_mask = threshold_feature_map(feature_map, method=threshold_method)
    
    # Resize prediction to match ground truth if needed
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
    
    # Binarize
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute metrics
    iou = compute_iou(pred_binary, gt_binary)
    precision, recall = compute_precision_recall(pred_binary, gt_binary)
    f1 = compute_f1(precision, recall)
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def process_single_image(
    image_path: str,
    output_dir: str,
    detectors: List = None,
    mask_path: Optional[str] = None,
    visualize: bool = True,
) -> Dict[str, Dict]:
    """Process a single image with all detectors."""
    # Load image
    image = load_image(image_path)
    image_name = Path(image_path).stem
    
    # Load mask if available
    gt_mask = None
    if mask_path and os.path.exists(mask_path):
        gt_mask = load_mask(mask_path)
    
    # Use default detectors if not specified
    if detectors is None:
        detectors = DEFAULT_DETECTORS
    
    results = {}
    
    for DetectorClass in detectors:
        detector = DetectorClass()
        print(f"  Running {detector.name}...")
        
        # Run detector
        feature_map = detector.detect(image)
        
        # Save feature map
        feature_path = os.path.join(output_dir, f"{image_name}_{detector.name}.png")
        save_image(feature_map, feature_path)
        
        # Evaluate if ground truth available
        metrics = None
        if gt_mask is not None:
            metrics = evaluate_detector(feature_map, gt_mask)
        
        # Create visualization
        if visualize:
            vis = visualize_result(image, feature_map, gt_mask)
            vis_path = os.path.join(output_dir, f"{image_name}_{detector.name}_vis.png")
            save_image(vis, vis_path)
        
        results[detector.name] = {
            'feature_path': feature_path,
            'metrics': metrics,
        }
    
    return results


def process_dataset(
    data_dir: str,
    output_dir: str,
    detectors: List = None,
) -> Dict[str, Dict]:
    """Process entire dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files
    files = get_dataset_files(data_dir)
    
    if not files:
        print(f"No images found in {data_dir}")
        return {}
    
    print(f"Found {len(files)} images in {data_dir}")
    
    all_results = {}
    
    for i, file_info in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing {file_info['name']}...")
        
        # Create output subdirectory for this image
        img_output = os.path.join(output_dir, file_info['name'])
        os.makedirs(img_output, exist_ok=True)
        
        results = process_single_image(
            image_path=file_info['image'],
            output_dir=img_output,
            detectors=detectors,
            mask_path=file_info.get('mask'),
        )
        
        all_results[file_info['name']] = results
    
    return all_results


def compute_summary_metrics(all_results: Dict) -> Dict:
    """Compute summary metrics across all images."""
    # Collect metrics per detector
    detector_metrics = {}
    
    for image_name, image_results in all_results.items():
        for detector_name, detector_result in image_results.items():
            if detector_result['metrics'] is None:
                continue
            
            if detector_name not in detector_metrics:
                detector_metrics[detector_name] = {
                    'iou': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                }
            
            for metric, value in detector_result['metrics'].items():
                detector_metrics[detector_name][metric].append(value)
    
    # Compute averages
    summary = {}
    for detector_name, metrics in detector_metrics.items():
        summary[detector_name] = {
            'iou_mean': float(np.mean(metrics['iou'])),
            'iou_std': float(np.std(metrics['iou'])),
            'precision_mean': float(np.mean(metrics['precision'])),
            'precision_std': float(np.std(metrics['precision'])),
            'recall_mean': float(np.mean(metrics['recall'])),
            'recall_std': float(np.std(metrics['recall'])),
            'f1_mean': float(np.mean(metrics['f1'])),
            'f1_std': float(np.std(metrics['f1'])),
            'num_images': len(metrics['iou']),
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Image Tamper Detection using Traditional Features'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to dataset directory (contains images/ and masks/)'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to single image to process'
    )
    parser.add_argument(
        '--mask', '-m',
        type=str,
        help='Path to ground truth mask (for single image mode)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--detectors',
        nargs='+',
        type=str,
        help='List of detector names to use (default: all primary detectors)'
    )
    parser.add_argument(
        '--list-detectors',
        action='store_true',
        help='List all available detectors and exit'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization output'
    )
    
    args = parser.parse_args()
    
    # List detectors
    if args.list_detectors:
        print("\nAvailable detectors:")
        print("\nPrimary detectors (10):")
        for det in DEFAULT_DETECTORS:
            print(f"  - {det.name}: {det.description}")
        print("\nAdvanced detectors:")
        for det in ALL_DETECTORS:
            if det not in DEFAULT_DETECTORS:
                print(f"  - {det.name}: {det.description}")
        return
    
    # Parse detector list
    detectors = None
    if args.detectors:
        detectors = []
        for name in args.detectors:
            det_class = get_detector_by_name(name)
            if det_class is None:
                print(f"Warning: Unknown detector '{name}', skipping")
            else:
                detectors.append(det_class)
        
        if not detectors:
            print("Error: No valid detectors specified")
            return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"Image Tamper Detection")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Process
    if args.image:
        # Single image mode
        all_results = {
            Path(args.image).stem: process_single_image(
                image_path=args.image,
                output_dir=args.output,
                detectors=detectors,
                mask_path=args.mask,
                visualize=not args.no_visualize,
            )
        }
    elif args.data:
        # Dataset mode
        all_results = process_dataset(
            data_dir=args.data,
            output_dir=args.output,
            detectors=detectors,
        )
    else:
        print("Error: Specify either --data or --image")
        parser.print_help()
        return
    
    # Compute summary
    summary = compute_summary_metrics(all_results)
    
    # Save results
    results_path = os.path.join(args.output, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'all_results': all_results,
            'summary': summary,
        }, f, indent=2)
    
    # Print summary
    if summary:
        print(f"\n{'='*60}")
        print("Summary Metrics:")
        print(f"{'='*60}")
        print(f"{'Detector':<20} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 60)
        for det_name, metrics in sorted(summary.items(), key=lambda x: x[1]['f1_mean'], reverse=True):
            print(f"{det_name:<20} {metrics['iou_mean']:>10.4f} {metrics['precision_mean']:>12.4f} "
                  f"{metrics['recall_mean']:>10.4f} {metrics['f1_mean']:>10.4f}")
    
    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {(datetime.now() - start_time).total_seconds():.2f}s")


if __name__ == '__main__':
    main()