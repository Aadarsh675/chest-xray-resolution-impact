import numpy as np
import cv2
import torch

from typing import List, Tuple, Union, Optional

from common import logger
from common.logger import get_logger

class DetectionProcessor:
    """
    Handles processing of detection results and bbox manipulation
    """

    def __init__(self, class_names, name_threshold_map, class_colors):
        """
        Initialize the detection processor

        Args:
            class_names: Dictionary mapping category IDs to names
            name_threshold_map: Dictionary mapping category IDs to score thresholds
            class_colors: Dictionary mapping class names to colors
            logger: Logger instance
        """
        self.class_names = class_names
        self.name_threshold_map = name_threshold_map
        self.class_colors = class_colors
        self.logger = get_logger(__name__)

    def extract_detections(
        self,
        mmdet_result,
        img_height: int,
        img_width: int,
        model_type: str,
        *,
        upper_percentage: float = 0.25,
        log_debug: bool = False,
    ):
        """
        Convert a single DetDataSample → list[dict] with richer debug logging.
        """
        upper_threshold = img_height * upper_percentage
        if log_debug:
            self.logger.debug(
                "[extract] model=%s  img=%dx%d  upper_thresh=%.1f",
                model_type, img_width, img_height, upper_threshold,
            )

        try:
            results = self.det_data_sample_to_mmdet2(mmdet_result)
        except Exception:
            self.logger.exception("[extract] det_data_sample_to_mmdet2 failed")
            return []

        if log_debug:
            # Pretty repr without flooding the log
            self.logger.debug("[extract] det_data_sample_to_mmdet2 returned a %s", type(results))
            if isinstance(results, tuple):
                self.logger.debug("   tuple len = %d (bbox, segm)", len(results))
            elif isinstance(results, list):
                self.logger.debug("   list len = %d (bbox only)", len(results))

        #  Unpack bbox / mask containers
        has_segm = isinstance(results, tuple) and len(results) >= 2
        bbox_results = results[0] if has_segm else results
        segm_results = results[1] if has_segm else None

        if has_segm:
            self.logger.info("[extract] masks PRESENT  |  classes=%d", len(segm_results))
        else:
            self.logger.warning("[extract] masks MISSING | treating as bbox-only output")

        #  Process bounding boxes
        detections = self._process_bbox_results(bbox_results)

        #  Attach segmentation masks (if any)
        if has_segm:
            try:
                if log_debug:
                    per_cls_pairs = [
                        (len(bbox_results[i]), len(segm_results[i])) for i in range(len(bbox_results))
                    ]
                    bad = [i for i, (b, s) in enumerate(per_cls_pairs) if b != s]
                    self.logger.debug("[extract] per-class (bboxes, masks) = %s", per_cls_pairs)
                    if bad:
                        self.logger.warning("[extract] classes with mismatch: %s", bad)

                self._add_segmentation_to_detections(detections, segm_results)
            except Exception:
                self.logger.exception("[extract] _add_segmentation_to_detections failed")

        #Stats before NMS
        counts = self._count_detections_by_category(detections)
        self._log_detection_counts(counts)

        # Category-wise NMS
        filtered = self._apply_nms_by_category(detections)

        return filtered

    def _process_bbox_results(self, results):
        """
        Process bbox results in standard format [x1, y1, x2, y2, score]

        Args:
            results: List of arrays, each containing detections for a specific class

        Returns:
            List of detection dictionaries
        """
        detections = []
        self.logger.info(f"Processing bounding box results")

        # Handle different result types
        if not isinstance(results, list):
            self.logger.error(f"Unexpected results format: {type(results)}. Expected list format.")
            return []

        # Process each class's detections
        for class_idx, class_detections in enumerate(results):
            self.logger.info(f"Processing class {class_idx + 1} with {len(class_detections)} detections")
            # Class ID is 1-indexed in our system
            cat_id = class_idx + 1
            category_name = self.class_names.get(cat_id, f"unknown_{cat_id}")
            threshold = self.name_threshold_map.get(cat_id, 0.5)

            # Skip if no detections for this class
            if not isinstance(class_detections, (list, np.ndarray)) or len(class_detections) == 0:
                self.logger.debug(f"No detections for class {cat_id} ({category_name})")
                continue

            self.logger.debug(f"Processing {len(class_detections)} detections for class {cat_id} ({category_name})")

            # Process each detection for this class
            detections_added = 0
            for detection in class_detections:
                try:
                    # Check if we have enough values in the detection array
                    if len(detection) >= 5:
                        x1, y1, x2, y2, score = detection[:5]

                        # Skip if confidence is below threshold
                        if score < threshold:
                            continue

                        # Ensure coordinates are properly ordered
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1

                        # Calculate width and height
                        width, height = x2 - x1, y2 - y1

                        # Skip if dimensions are too small
                        if width < 5 or height < 5:
                            continue

                        # Add to detections
                        detections.append({
                            "bbox": [int(x1), int(y1), int(width), int(height)],
                            "score": float(score),
                            "category_id": cat_id,
                            "category_name": category_name,
                        })
                        detections_added += 1
                except Exception as e:
                    self.logger.warning(f"Error processing detection for class {cat_id}: {e}")
                    continue

            self.logger.debug(f"Added {detections_added} detections for class {cat_id} ({category_name})")

        return detections

    def _add_segmentation_to_detections(self, detections, segm_results):
        """
        Add segmentation data to detections if available

        Args:
            detections: List of detection dictionaries
            segm_results: Segmentation results
        """
        self.logger.info("Processing segmentation data")

        # Create a map of detections by class and index for easier lookup
        detection_map = {}
        for i, det in enumerate(detections):
            cat_id = det['category_id']
            if cat_id not in detection_map:
                detection_map[cat_id] = {}
            # Use the detection index in original list as the key
            detection_map[cat_id][i] = det

        # Process segmentation results for each class
        for class_idx, class_segmentations in enumerate(segm_results):
            cat_id = class_idx + 1
            category_name = self.class_names.get(cat_id, f"unknown_{cat_id}")

            # Skip if no segmentations for this class or no detections to match
            if cat_id not in detection_map or not class_segmentations:
                self.logger.debug(f"Skipping segmentations for class {cat_id} ({category_name})")
                continue

            self.logger.debug(f"Processing segmentations for class {cat_id} ({category_name})")

            # Match segmentations to detections by index
            for i, segm in enumerate(class_segmentations):
                try:
                    # Find corresponding detection
                    matching_dets = [det_idx for det_idx in detection_map[cat_id].keys()]
                    if i < len(matching_dets):
                        det_idx = matching_dets[i]
                        det = detection_map[cat_id][det_idx]

                        # Add segmentation to detection
                        if isinstance(segm, dict):  # RLE format
                            det["segmentation"] = segm
                        elif isinstance(segm, list):  # Polygon format
                            det["segmentation"] = [segm]
                        elif isinstance(segm, np.ndarray):  # Binary mask
                            # Convert binary mask to polygon
                            contours, _ = cv2.findContours(
                                segm.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            polygons = []
                            for contour in contours:
                                # Convert contour to polygon format
                                polygon = contour.flatten().tolist()
                                if len(polygon) >= 6:  # At least 3 points
                                    polygons.append(polygon)
                            if polygons:
                                det["segmentation"] = polygons
                except Exception as e:
                    self.logger.warning(f"Error processing segmentation for class {cat_id}, index {i}: {e}")
                    continue

    def det_data_sample_to_mmdet2(
        self,
        det_sample,
        *,
        num_classes: Optional[int] = None,
        include_masks: bool = True
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[List]]]:
        """
        Convert an MMDetection 3.x `DetDataSample` into the MMDetection 2.x
        inference-output format used by legacy pipelines.

        Parameters
        ----------
        det_sample : mmdet.structures.DetDataSample
            The prediction returned by `model.test_step()` or `inference_detector`
            in MMDet 3.  Must contain ``pred_instances``.
        num_classes : int, optional
            Total number of categories in the model.  If *None*, the value is
            inferred from :pyattr:`pred_instances.labels`.  Supplying it avoids an
            ``O(num_classes)`` scan and guarantees a fixed-length list (helpful if a
            category has zero detections).
        include_masks : bool, default ``True``
            If the sample has ``pred_instances.masks`` **and** this flag is ``True``,
            the function returns **``(bbox_results, segm_results)``**; otherwise
            only ``bbox_results`` (the list of bbox arrays) is returned.

        Returns
        -------
        bbox_results : List[np.ndarray]
            Length == ``num_classes``.  Each element is an ``(Ni, 5)`` ndarray
            ``[x1, y1, x2, y2, score]`` (float32).  Classes with no detections get
            ``np.empty((0, 5), dtype=np.float32)``.
        segm_results : List[List]  (only if ``include_masks`` is ``True`` and masks exist)
            Length == ``num_classes``.  Each element is a list of binary masks or
            COCO-style RLEs that line up with the rows of the corresponding bbox
            array.

        Notes
        -----
        *  Works with both *BitmapMasks* and *RLEMasks*; the objects are passed
           through unchanged.
        *  The function does **not** perform any additional NMS or score filtering;
           it merely reshapes the data.
        """
        pred = det_sample.pred_instances

        # Pull the raw tensors / arrays
        bboxes  = pred.bboxes.cpu().numpy()           # (N, 4)
        scores  = pred.scores.cpu().numpy()           # (N,)
        labels  = pred.labels.cpu().numpy()           # (N,)
        has_masks = (
            include_masks
            and pred is not None
            and hasattr(pred, 'masks')
        )
        ### UNCOMMENT FOR TESTING
        # self.logger.debug(
        #            "bboxes:  shape=%s, dtype=%s, first=%s",
        #            bboxes.shape, bboxes.dtype, bboxes[:1],
        #        )
        # self.logger.debug(
        #     "scores:  shape=%s, dtype=%s, min=%.4f, max=%.4f",
        #     scores.shape, scores.dtype,
        #     scores.min() if scores.size else float("nan"),
        #     scores.max() if scores.size else float("nan"),
        # )
        # self.logger.debug(
        #     "labels:  shape=%s, dtype=%s, unique=%s",
        #     labels.shape, labels.dtype, np.unique(labels),
        # )
        # self.logger.debug("has_masks = %s", has_masks)


        if num_classes is None:
            num_classes = int(labels.max()) + 1 if labels.size else 0

        # Initialise per-class containers
        bbox_results: List[np.ndarray] = [
            np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)
        ]
        if has_masks:
            self.logger.debug("Results has a mask.")
            segm_results: List[List] = [[] for _ in range(num_classes)]

        # Slice detections per class
        for cls in range(num_classes):
            cls_idxs = np.where(labels == cls)[0]
            if cls_idxs.size == 0:
                continue

            # Stack [x1,y1,x2,y2,score]
            cls_bboxes = np.hstack([bboxes[cls_idxs], scores[cls_idxs, None]]).astype(np.float32)
            bbox_results[cls] = cls_bboxes

            if has_masks:
                raw_masks = pred.masks[cls_idxs]  # BitmapMasks / RLEMasks / Tensor
                cpu_masks: List = []

                for m in raw_masks:
                    # (a) Plain torch.Tensor mask                             ──────
                    if isinstance(m, torch.Tensor):
                        cpu_masks.append(m.detach().cpu().numpy())

                    # (b) BitmapMasks / RLEMasks objects (MMDet 3.x)          ──────
                    #     They expose .to_tensor() (Bitmap) or .to_ndarray()
                    elif hasattr(m, "to_tensor"):
                        cpu_masks.append(m.to_tensor(dtype=torch.uint8, device="cpu").numpy())
                    elif hasattr(m, "to_ndarray"):
                        cpu_masks.append(m.to_ndarray())

                    # (c) Already a NumPy array                               ──────
                    elif isinstance(m, np.ndarray):
                        cpu_masks.append(m)

                    # (d) COCO-style RLE dict / polygon list – already on CPU ──────
                    else:
                        cpu_masks.append(m)
                segm_results[cls] = cpu_masks

        return (bbox_results, segm_results) if has_masks else bbox_results

    def _count_detections_by_category(self, detections):
        """Count detections by category"""
        detection_counts = {}
        for det in detections:
            cat_id = det['category_id']
            if cat_id not in detection_counts:
                detection_counts[cat_id] = 0
            detection_counts[cat_id] += 1
        return detection_counts

    def _log_detection_counts(self, detection_counts):
        """Log detection counts by category"""
        for cat_id in sorted(detection_counts.keys()):
            category_name = self.class_names.get(cat_id, f"unknown_{cat_id}")
            self.logger.info(f"Class {cat_id} ({category_name}): {detection_counts[cat_id]} detections")

    def _apply_nms_by_category(self, detections):
        """Apply NMS to detections by category"""
        filtered_detections = []
        categories = set(d['category_id'] for d in detections)

        for category in categories:
            category_detections = [d for d in detections if d['category_id'] == category]
            filtered_category = self.nms(category_detections)
            category_name = self.class_names.get(category, f"unknown_{category}")
            self.logger.info(f"NMS for class {category} ({category_name}): {len(category_detections)} -> {len(filtered_category)}")
            filtered_detections.extend(filtered_category)

        return filtered_detections

    def _log_results_structure(self, results):
        """Log the structure of results for debugging"""
        if isinstance(results, list):
            self.logger.error(f"Results is a list of {len(results)} elements")
            for i, item in enumerate(results):
                if hasattr(item, 'shape'):
                    self.logger.error(f"  results[{i}].shape = {item.shape}")
                else:
                    self.logger.error(f"  results[{i}] type = {type(item)}")

    def nms(self, detections, iou_threshold=0.75):
        """
        Perform non-maximum suppression on detections

        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered list of detection dictionaries
        """
        if not detections:
            return []

        # Convert bounding boxes to numpy arrays
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        indices = np.argsort(scores)[::-1]

        keep = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            if indices.size == 1:
                break

            ious = []
            for j in indices[1:]:
                iou = self.compute_iou(
                    [boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]],
                    [boxes[j][0], boxes[j][1], boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]]
                )
                ious.append(iou)
            ious = np.array(ious)
            indices = indices[1:][ious < iou_threshold]

        return [detections[i] for i in keep]

    def compute_iou(self, box1, box2):
        """
        Compute IoU between two bounding boxes

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def format_detections(self, detections):
        """
        Format detections for result message

        Args:
            detections: List of detection dictionaries

        Returns:
            List of formatted detection dictionaries
        """
        formatted_detections = []
        for det in detections:
            formatted_det = {
                "class": det['category_name'],
                "confidence": det['score'],
                "bounding_box": det['bbox']
            }

            # Add segmentation if it exists
            if 'segmentation' in det:
                formatted_det["segmentation"] = det['segmentation']

            formatted_detections.append(formatted_det)

        return formatted_detections
