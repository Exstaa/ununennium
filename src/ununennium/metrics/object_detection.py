"""Object detection metrics.

This module provides standard object detection evaluation metrics including:
- IoU (Intersection over Union) between bounding boxes
- True Positives, False Positives, False Negatives per class
- Precision and Recall per class
- Average Precision (AP) per class
- Mean Average Precision (mAP) across all classes
- AP50 (AP at IoU threshold 0.5)

All computations use PyTorch tensors for GPU acceleration.
"""

from __future__ import annotations

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: Tensor of shape (N, 4) with boxes in [x1, y1, x2, y2] format.
        boxes2: Tensor of shape (M, 4) with boxes in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M) where element [i, j] is IoU between
        boxes1[i] and boxes2[j].

    Example:
        >>> pred_boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32)
        >>> gt_boxes = torch.tensor([[12, 12, 48, 48]], dtype=torch.float32)
        >>> iou_matrix = box_iou(pred_boxes, gt_boxes)
        >>> iou_matrix.shape
        torch.Size([2, 1])
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection coordinates
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    # Compute intersection area
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute union
    union = area1[:, None] + area2[None, :] - intersection

    # Compute IoU
    iou = intersection / union.clamp(min=1e-6)

    return iou


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match predicted boxes to ground truth boxes.

    For each prediction, determines if it is a true positive (TP) or
    false positive (FP) based on IoU threshold and class matching.

    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Confidence scores for predictions (N,).
        pred_labels: Class labels for predictions (N,).
        gt_boxes: Ground truth boxes (M, 4) in [x1, y1, x2, y2] format.
        gt_labels: Class labels for ground truth (M,).
        iou_threshold: IoU threshold for positive match.

    Returns:
        Tuple of (tp, fp, matched_gt_indices):
        - tp: Boolean tensor (N,) indicating true positives.
        - fp: Boolean tensor (N,) indicating false positives.
        - matched_gt_indices: Index of matched GT box for each prediction (-1 if no match).

    Example:
        >>> pred_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        >>> pred_scores = torch.tensor([0.9])
        >>> pred_labels = torch.tensor([1])
        >>> gt_boxes = torch.tensor([[12, 12, 48, 48]], dtype=torch.float32)
        >>> gt_labels = torch.tensor([1])
        >>> tp, fp, _ = match_predictions(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    """
    device = pred_boxes.device
    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]

    tp = torch.zeros(num_preds, dtype=torch.bool, device=device)
    fp = torch.zeros(num_preds, dtype=torch.bool, device=device)
    matched_gt_indices = torch.full((num_preds,), -1, dtype=torch.long, device=device)

    if num_preds == 0:
        return tp, fp, matched_gt_indices

    if num_gts == 0:
        fp[:] = True
        return tp, fp, matched_gt_indices

    # Sort predictions by confidence (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)

    # Compute IoU matrix
    iou_matrix = box_iou(pred_boxes, gt_boxes)  # (N, M)

    # Track which GT boxes have been matched
    gt_matched = torch.zeros(num_gts, dtype=torch.bool, device=device)

    for pred_idx in sorted_indices:
        pred_label = pred_labels[pred_idx]

        # Find GT boxes with matching class
        class_mask = gt_labels == pred_label
        if not class_mask.any():
            fp[pred_idx] = True
            continue

        # Get IoUs for this prediction with matching class GTs
        ious = iou_matrix[pred_idx] * class_mask.float()
        ious[gt_matched] = 0  # Exclude already matched GTs

        # Find best matching GT
        max_iou, max_idx = ious.max(dim=0)

        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[pred_idx] = True
            gt_matched[max_idx] = True
            matched_gt_indices[pred_idx] = max_idx
        else:
            fp[pred_idx] = True

    return tp, fp, matched_gt_indices


def compute_tp_fp_fn(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute True Positives, False Positives, and False Negatives per class.

    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Confidence scores for predictions (N,).
        pred_labels: Class labels for predictions (N,).
        gt_boxes: Ground truth boxes (M, 4) in [x1, y1, x2, y2] format.
        gt_labels: Class labels for ground truth (M,).
        num_classes: Total number of classes.
        iou_threshold: IoU threshold for positive match.

    Returns:
        Tuple of (tp_per_class, fp_per_class, fn_per_class), each of shape (num_classes,).

    Example:
        >>> tp, fp, fn = compute_tp_fp_fn(pred_boxes, pred_scores, pred_labels,
        ...                                gt_boxes, gt_labels, num_classes=10)
    """
    device = pred_boxes.device if pred_boxes.numel() > 0 else gt_boxes.device

    tp_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    fp_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    fn_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    # Match predictions to ground truth
    tp, fp, _ = match_predictions(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold
    )

    # Count TP and FP per class
    for c in range(num_classes):
        class_mask = pred_labels == c
        tp_per_class[c] = tp[class_mask].sum()
        fp_per_class[c] = fp[class_mask].sum()

        # Count FN: GT boxes not matched
        gt_class_count = (gt_labels == c).sum()
        fn_per_class[c] = gt_class_count - tp_per_class[c]

    return tp_per_class, fp_per_class, fn_per_class


def precision_recall(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute precision and recall per class.

    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Confidence scores for predictions (N,).
        pred_labels: Class labels for predictions (N,).
        gt_boxes: Ground truth boxes (M, 4) in [x1, y1, x2, y2] format.
        gt_labels: Class labels for ground truth (M,).
        num_classes: Total number of classes.
        iou_threshold: IoU threshold for positive match.

    Returns:
        Tuple of (precision_per_class, recall_per_class), each of shape (num_classes,).

    Example:
        >>> precision, recall = precision_recall(pred_boxes, pred_scores, pred_labels,
        ...                                       gt_boxes, gt_labels, num_classes=10)
    """
    tp, fp, fn = compute_tp_fp_fn(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes, iou_threshold
    )

    tp = tp.float()
    fp = fp.float()
    fn = fn.float()

    precision = tp / (tp + fp).clamp(min=1e-6)
    recall = tp / (tp + fn).clamp(min=1e-6)

    # Handle classes with no predictions or ground truth
    no_preds = (tp + fp) == 0
    no_gts = (tp + fn) == 0
    precision[no_preds] = 0.0
    recall[no_gts] = 0.0

    return precision, recall


def _compute_ap_single_class(
    pred_scores: torch.Tensor,
    pred_is_tp: torch.Tensor,
    num_gt: int,
) -> torch.Tensor:
    """Compute Average Precision for a single class using 101-point interpolation.

    Args:
        pred_scores: Confidence scores for predictions of this class (K,).
        pred_is_tp: Boolean tensor indicating if each prediction is a TP (K,).
        num_gt: Number of ground truth boxes for this class.

    Returns:
        Average Precision (scalar tensor).
    """
    device = pred_scores.device

    if num_gt == 0:
        return torch.tensor(0.0, device=device)

    if pred_scores.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Sort by confidence descending
    sorted_indices = torch.argsort(pred_scores, descending=True)
    tp_sorted = pred_is_tp[sorted_indices].float()

    # Compute cumulative TP and FP
    tp_cumsum = torch.cumsum(tp_sorted, dim=0)
    fp_cumsum = torch.cumsum(1 - tp_sorted, dim=0)

    # Compute precision and recall at each threshold
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt

    # Prepend (0, 1) for recall=0
    precision = torch.cat([torch.ones(1, device=device), precision])
    recall = torch.cat([torch.zeros(1, device=device), recall])

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = torch.max(precision[i], precision[i + 1])

    # 101-point interpolation (COCO style)
    recall_thresholds = torch.linspace(0, 1, 101, device=device)
    ap = torch.tensor(0.0, device=device)

    for t in recall_thresholds:
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()

    ap /= 101

    return ap


def average_precision_per_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute Average Precision (AP) for each class.

    Uses 101-point interpolation (COCO-style) for AP computation.

    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Confidence scores for predictions (N,).
        pred_labels: Class labels for predictions (N,).
        gt_boxes: Ground truth boxes (M, 4) in [x1, y1, x2, y2] format.
        gt_labels: Class labels for ground truth (M,).
        num_classes: Total number of classes.
        iou_threshold: IoU threshold for positive match.

    Returns:
        AP per class of shape (num_classes,).

    Example:
        >>> ap_per_class = average_precision_per_class(
        ...     pred_boxes, pred_scores, pred_labels,
        ...     gt_boxes, gt_labels, num_classes=10, iou_threshold=0.5
        ... )
    """
    device = pred_boxes.device if pred_boxes.numel() > 0 else gt_boxes.device
    ap_per_class = torch.zeros(num_classes, device=device)

    # Match all predictions
    tp, fp, _ = match_predictions(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold
    )

    for c in range(num_classes):
        # Get predictions for this class
        class_mask = pred_labels == c
        class_scores = pred_scores[class_mask]
        class_tp = tp[class_mask]

        # Count GT for this class
        num_gt = (gt_labels == c).sum().item()

        ap_per_class[c] = _compute_ap_single_class(class_scores, class_tp, num_gt)

    return ap_per_class


def mean_average_precision(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> torch.Tensor:
    """Compute mean Average Precision (mAP) across classes and IoU thresholds.

    By default computes mAP@[0.5:0.95:0.05] (COCO-style mAP).

    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Confidence scores for predictions (N,).
        pred_labels: Class labels for predictions (N,).
        gt_boxes: Ground truth boxes (M, 4) in [x1, y1, x2, y2] format.
        gt_labels: Class labels for ground truth (M,).
        num_classes: Total number of classes.
        iou_thresholds: List of IoU thresholds. Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        Mean AP (scalar tensor).

    Example:
        >>> mAP = mean_average_precision(
        ...     pred_boxes, pred_scores, pred_labels,
        ...     gt_boxes, gt_labels, num_classes=10
        ... )
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    device = pred_boxes.device if pred_boxes.numel() > 0 else gt_boxes.device
    aps = []

    for iou_thresh in iou_thresholds:
        ap = average_precision_per_class(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes, iou_thresh
        )
        # Average over classes with GT
        classes_with_gt = torch.tensor(
            [(gt_labels == c).any() for c in range(num_classes)], device=device
        )
        if classes_with_gt.any():
            aps.append(ap[classes_with_gt].mean())
        else:
            aps.append(torch.tensor(0.0, device=device))

    return torch.stack(aps).mean()


def ap50(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute AP at IoU threshold 0.5 (AP50).

    This is the traditional PASCAL VOC metric.

    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
        pred_scores: Confidence scores for predictions (N,).
        pred_labels: Class labels for predictions (N,).
        gt_boxes: Ground truth boxes (M, 4) in [x1, y1, x2, y2] format.
        gt_labels: Class labels for ground truth (M,).
        num_classes: Total number of classes.

    Returns:
        Mean AP at IoU=0.5 (scalar tensor).

    Example:
        >>> ap_50 = ap50(pred_boxes, pred_scores, pred_labels,
        ...              gt_boxes, gt_labels, num_classes=10)
    """
    ap = average_precision_per_class(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes, iou_threshold=0.5
    )

    # Average over classes with GT
    device = pred_boxes.device if pred_boxes.numel() > 0 else gt_boxes.device
    classes_with_gt = torch.tensor(
        [(gt_labels == c).any() for c in range(num_classes)], device=device
    )

    if classes_with_gt.any():
        return ap[classes_with_gt].mean()
    return torch.tensor(0.0, device=device)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Object detection evaluation with 3 classes

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ground truth boxes and labels (M=5 boxes)
    gt_boxes = torch.tensor(
        [
            [10, 10, 50, 50],  # Class 0
            [100, 100, 150, 150],  # Class 1
            [200, 200, 250, 250],  # Class 1
            [300, 50, 350, 100],  # Class 2
            [400, 400, 450, 450],  # Class 0
        ],
        dtype=torch.float32,
        device=device,
    )
    gt_labels = torch.tensor([0, 1, 1, 2, 0], device=device)

    # Predicted boxes, scores, and labels (N=6 predictions)
    pred_boxes = torch.tensor(
        [
            [12, 12, 48, 48],  # Good match for GT[0], class 0
            [105, 105, 145, 145],  # Good match for GT[1], class 1
            [210, 210, 245, 245],  # Good match for GT[2], class 1
            [500, 500, 550, 550],  # False positive, class 1
            [305, 55, 345, 95],  # Good match for GT[3], class 2
            [50, 50, 100, 100],  # False positive, class 0
        ],
        dtype=torch.float32,
        device=device,
    )
    pred_scores = torch.tensor([0.95, 0.88, 0.75, 0.60, 0.92, 0.45], device=device)
    pred_labels = torch.tensor([0, 1, 1, 1, 2, 0], device=device)

    num_classes = 3

    print("=" * 60)
    print("Object Detection Metrics Example")
    print("=" * 60)

    # 1. Compute box IoU
    print("\n1. Box IoU Matrix (pred vs gt):")
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    print(f"   Shape: {iou_matrix.shape}")
    print(f"   Max IoU per prediction: {iou_matrix.max(dim=1).values}")

    # 2. Compute TP, FP, FN per class
    print("\n2. TP/FP/FN per class (IoU=0.5):")
    tp, fp, fn = compute_tp_fp_fn(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes
    )
    for c in range(num_classes):
        print(f"   Class {c}: TP={tp[c].item()}, FP={fp[c].item()}, FN={fn[c].item()}")

    # 3. Compute Precision and Recall per class
    print("\n3. Precision/Recall per class:")
    prec, rec = precision_recall(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes
    )
    for c in range(num_classes):
        print(f"   Class {c}: Precision={prec[c]:.4f}, Recall={rec[c]:.4f}")

    # 4. Compute AP per class at IoU=0.5
    print("\n4. Average Precision (AP) per class at IoU=0.5:")
    ap_per_class = average_precision_per_class(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes, iou_threshold=0.5
    )
    for c in range(num_classes):
        print(f"   Class {c}: AP={ap_per_class[c]:.4f}")

    # 5. Compute AP50
    print("\n5. AP50 (mean AP at IoU=0.5):")
    ap_50 = ap50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes)
    print(f"   AP50 = {ap_50:.4f}")

    # 6. Compute mAP (COCO-style)
    print("\n6. mAP (COCO-style, IoU=[0.5:0.95:0.05]):")
    mAP = mean_average_precision(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes
    )
    print(f"   mAP = {mAP:.4f}")

    print("\n" + "=" * 60)
