import os
import cv2
import sys
import torch
import json
import tempfile
import time
import traceback

from datetime import datetime
from mmdet.apis import init_detector, inference_detector

from common.aws import download_image_from_s3, upload_frame_to_s3, parse_s3_path, get_thread_local_s3_transfer
from common.rabbitmq import get_rabbitmq_channel, publish_results_message
from common.logger import get_logger

from configs.detection_config import get_config_paths, get_queue_name
from services.detection.src.utils import load_json_file, load_class_colors, resize_image
from services.detection.src.detection import DetectionProcessor
from services.detection.src.visualization import Visualizer

log = get_logger(__name__)
def worker_process(args, gpu_id, timeout):
    """
    Worker process function for handling inference requests

    Args:
        args: Command line arguments
        gpu_id: GPU ID for this worker
        timeout: Timeout duration in seconds
        logger: Logger instance
    """
    try:
        # Set device for this worker
        num_gpus = torch.cuda.device_count()
        gpu_id = gpu_id % num_gpus
        args.device = f'cuda:{gpu_id}'

        # Get RabbitMQ channel
        channel = get_rabbitmq_channel()

        # Create and run the inference worker
        worker = InferenceWorker(args, channel, timeout)
        worker.run()
    except KeyboardInterrupt:
                log.warning(f"[GPU {gpu_id}] Interrupted — shutting down worker.")
                # clean up open files / S3 transfers if needed
                sys.exit(0)

class InferenceWorker:
    """
    Worker for handling inference requests
    """

    def __init__(self, args, channel, timeout):
        """
        Initialize the inference worker

        Args:
            args: Command line arguments
            channel: RabbitMQ channel
            timeout: Timeout duration in seconds
            logger: Logger instance
        """
        self.args = args
        self.channel = channel
        self.timeout = timeout
        self.logger = log

        # Initialize stats
        self.total_images = 0
        self.total_time = 0
        self.start_time = time.time()
        self.logger.debug("Entered the initialization")
        # Load configurations
        self._load_configurations()

        # Initialize model
        self.model = init_detector(self.args.config, self.args.checkpoint, device=self.args.device)
        self.logger.info("Model initialized")

        # Initialize detection processor
        self.detection_processor = DetectionProcessor(
            self.class_names,
            self.name_threshold_map,
            self.class_colors
        )

        # Initialize visualizer
        self.visualizer = Visualizer(
            self.class_colors,
            self.args.model,
            self.logger
        )

        # Determine which RabbitMQ queue to use based on model and dataset
        self.queue_name = get_queue_name(self.args.model, self.args.dataset, self.logger)
        self.logger.info(f'InferenceWorker initialized - using queue: {self.queue_name}')

        # GPU availability checking raises an error if unavailable
        if not torch.cuda.is_available() or 'cuda' not in self.args.device:
            raise RuntimeError("GPU is not available or device is not set to CUDA.")
        else:
            self.gpu_id = int(self.args.device.split(':')[1])


    def _load_configurations(self):
        """
        Load configuration files
        """
        # Get configuration paths
        config_paths = get_config_paths(self.args.model, self.args.dataset)

        # Load configurations
        db_id_data = load_json_file(config_paths['db_id'])
        self.class_names = {c['id']: c['name'] for c in db_id_data}
        self.id_db_map = {c['id']: c['db_id'] for c in db_id_data}
        self.name_threshold_map = {c['id']: c['threshold'] for c in db_id_data}
        self.class_colors = load_class_colors(config_paths['class_color'])

    def infer(self, image_path):
        """
        Run inference on an image

        Args:
            image_path: Path to the input image

        Returns:
            Detection results
        """
        if image_path is None:
            raise ValueError("image_path is None")

        # Check if the file exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")

        return inference_detector(self.model, image_path)

    def run(self):
        """
        Main worker loop with enhanced logging
        """
        # Set prefetch count and record start time
        self.channel.basic_qos(prefetch_count=1)
        self.start_time = time.time()
        worker_id = f"{self.args.device}-{os.getpid()}"

        # Log detailed worker startup information
        self.logger.info(f"[{worker_id}] WORKER STARTED | Model: {self.args.model} | Dataset: {self.args.dataset} | Queue: {self.queue_name}")
        self.logger.info(f"[{worker_id}] Model config: {os.path.basename(self.args.config)} | Checkpoint: {os.path.basename(self.args.checkpoint)}")

        # Initialize statistics counters
        idle_cycles = 0
        last_stats_time = time.time()
        stats_interval = 60  # Log stats every minute
        processed_since_last_stats = 0

        while True:
            try:
                # Get a message from the queue
                method_frame, header_frame, body = self.channel.basic_get(self.queue_name)

                if method_frame:
                    # Reset idle counter when message is received
                    idle_cycles = 0

                    # Acknowledge the message
                    self.channel.basic_ack(method_frame.delivery_tag)

                    # Parse the message body
                    data = json.loads(body)
                    job_id = data.get('job_id')
                    s3_frame_path = data.get('s3_frame_path')
                    video_id = data.get('video_id')
                    frame_id = data.get('frame_id', 'unknown')
                    frame_number = data.get('frame_number', 'unknown')

                    # Create a job context for consistent logging
                    job_ctx = f"[{worker_id}][Job:{job_id}][Frame:{frame_id}/{frame_number}]"

                    # Log job receipt
                    self.logger.info(f"{job_ctx} RECEIVED | Video: {video_id} | S3 Path: {s3_frame_path}")

                    # Extract folder path from s3_frame_path for organizing files
                    folder_path = video_id

                    # Download the image with timing
                    self.logger.info(f"{job_ctx} Downloading image from S3...: {s3_frame_path} | Folder: {folder_path}")
                    download_start = time.time()
                    image_path = download_image_from_s3(s3_frame_path, folder_path)
                    download_time = time.time() - download_start

                    if not image_path:
                        self.logger.error(f"{job_ctx} FAILED | Could not download image from S3")
                        self.publish_failure_result(data, "Failed to download image")
                        continue

                    self.logger.debug(f"{job_ctx} Downloaded image in {download_time:.3f}s | Path: {image_path}")

                    # Resize the image with timing
                    resize_start = time.time()
                    resize_image(image_path, image_path, 1920, 1080, self.logger)
                    resize_time = time.time() - resize_start
                    self.logger.debug(f"{job_ctx} Resized image in {resize_time:.3f}s to 1920x1080")

                    # Log GPU memory before inference if using GPU
                    self.logger.debug(f"{job_ctx} Pre-inference GPU memory: {torch.cuda.memory_allocated(self.gpu_id) / 1024**2:.2f} MB")

                    # Run inference with timing
                    start_inference_time = time.time()
                    results = self.infer(image_path)
                    inference_time = time.time() - start_inference_time

                    # Update statistics
                    self.total_images += 1
                    self.total_time += inference_time
                    processed_since_last_stats += 1

                    # Log inference completion
                    self.logger.info(f"{job_ctx} INFERENCE COMPLETE | Time: {inference_time:.3f}s | Avg: {self.total_time/self.total_images:.3f}s")

                    # Log GPU memory after inference if using GPU
                    self.logger.debug(f"{job_ctx} Post-inference GPU memory: {torch.cuda.memory_allocated(self.gpu_id) / 1024**2:.2f} MB")

                    try:
                        self.process_detections(results, data, image_path, image_path, folder_path)
                        self.logger.info(f"{job_ctx} PROCESSING COMPLETE | Total job time: {time.time() - start_inference_time:.3f}s")
                    except Exception as e:
                        self.logger.error(f"{job_ctx} PROCESSING FAILED | Error: {str(e)}", exc_info=True)

                    # Reset start time for timeout calculation
                    self.start_time = time.time()

                else:
                    # Increment idle counter
                    idle_cycles += 1

                    # Check for timeout
                    if time.time() - self.start_time > self.timeout:
                        self.logger.info(f"[{worker_id}] TIMEOUT | No messages in queue '{self.queue_name}' for {self.timeout}s | Exiting")
                        # Print final summary before exiting
                        self.print_summary()
                        break

                    # Log periodic idle messages (every 20 cycles = ~10 seconds)
                    if idle_cycles % 20 == 0:
                        self.logger.debug(f"[{worker_id}] Idle for {(time.time() - self.start_time):.1f}s | Waiting for messages in '{self.queue_name}'")

                    time.sleep(0.5)  # Small delay to avoid tight polling

                # Log periodic statistics
                current_time = time.time()
                if current_time - last_stats_time > stats_interval:
                    # Only log if we've processed at least one image since last stats
                    if processed_since_last_stats > 0:
                        throughput = processed_since_last_stats / (current_time - last_stats_time)
                        self.logger.info(f"[{worker_id}] STATS | Processed: {processed_since_last_stats} frames | "
                                        f"Throughput: {throughput:.2f} fps | "
                                        f"Total processed: {self.total_images}")

                        # Reset counters
                        processed_since_last_stats = 0
                        last_stats_time = current_time

                        # Log GPU utilization if available
                        self.logger.info(f"[{worker_id}] GPU memory: {torch.cuda.memory_allocated(self.gpu_id) / 1024**2:.2f} MB / "
                                            f"{torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**2:.2f} MB")

            except Exception as e:
                # Enhanced error logging
                error_id = f"ERR-{int(time.time())}"
                self.logger.error(f"[{worker_id}][{error_id}] CRITICAL ERROR in worker loop: {str(e)}", exc_info=True)

                # Try to get traceback information

                trace_str = traceback.format_exc()
                self.logger.error(f"[{worker_id}][{error_id}] Traceback: {trace_str}")

                # Publish failure result if we have job data
                if 'data' in locals() and data:
                    job_id = data.get('job_id', 'unknown')
                    self.logger.error(f"[{worker_id}][{error_id}][Job:{job_id}] Publishing failure result")
                    self.publish_failure_result(data, f"Internal error: {error_id} - {str(e)}")

                # Small delay to prevent rapid error cycles
                time.sleep(1)

        # Final log message
        self.logger.info(f"[{worker_id}] WORKER STOPPED | Model: {self.args.model} | Dataset: {self.args.dataset}")
        self.print_summary()

    def process_detections(self, results, data, image_name, image_path, folder_path):
        """
        Process detection results and publish to results queue

        Args:
            results: Detection results from the model
            data: Original job data from the orchestrator
            image_name: Name of the image
            image_path: Path to the image
            folder_path: Folder path for organization
        """
        # Extract all required fields from input data
        job_id = data.get('job_id')
        video_id = data.get('video_id')
        frame_id = data.get('frame_id')
        frame_number = data.get('frame_number')
        parent_job_id = data.get('parent_job_id')
        s3_frame_path = data.get('s3_frame_path')

        # Get the height and width of the image
        img_height, img_width = cv2.imread(image_path).shape[:2]

        # Extract detections from the results
        detections = self.detection_processor.extract_detections(
            results, img_height, img_width, self.args.model
        )

        # Overlay detections on the image
        output_path = self.visualizer.overlay_detections(image_path, detections)

        # Upload the image with detections to S3
        s3_detection_path = self.upload_frame(output_path, s3_frame_path)

        # Create COCO JSON from detections
        coco_json = self.create_coco_json(detections, image_path, img_width, img_height)

        # Upload COCO JSON to S3
        s3_coco_json_path = self.upload_coco_json(coco_json, s3_frame_path)

        self.logger.info(f"Image processed for job {job_id} with model {self.args.model}")
        self.logger.info(f"COCO JSON uploaded to {s3_coco_json_path}")

        # Format detections for the orchestrator
        formatted_detections = self.detection_processor.format_detections(detections)

        # Create result message for the workflow orchestrator
        result_message = {
            "job_id": job_id,
            "status": "completed",
            "job_type": "detection",
            "video_id": video_id,
            "frame_id": frame_id,
            "frame_number": frame_number,
            "parent_job_id": parent_job_id,
            "s3_frame_path": s3_detection_path,
            "s3_coco_json_path": s3_coco_json_path,
            "timestamp": datetime.now().isoformat(),
            "results": {
                "detections": formatted_detections
            },
            "model": self.args.model,
            "dataset": self.args.dataset
        }

        # Use the publish_results_message function to send to the results queue
        publish_results_message(self.channel, result_message)
        self.logger.info(f"Published detection results for job {job_id}")

    def create_coco_json(self, detections, image_path, img_width, img_height):
        """
        Create a COCO format JSON from detections

        Args:
            detections: List of detection dictionaries
            image_path: Path to the image
            img_width: Image width
            img_height: Image height

        Returns:
            Dictionary in COCO JSON format
        """
        coco_json = {
            "images": [{
                "id": 1,
                "width": img_width,
                "height": img_height,
                "file_name": image_path
            }],
            "annotations": [],
            "categories": []
        }

        categories_set = set()
        annotation_id = 1

        for detection in detections:
            category_id = detection.get('category_id')
            category_name = detection.get('category_name')
            bbox = detection.get('bbox')
            score = detection.get('score')

            # Skip if essential information is missing
            if category_id is None or bbox is None:
                self.logger.warning(f"Skipping detection due to missing category_id or bbox: {detection}")
                continue

            # Add category if not already present
            if category_id not in categories_set:
                coco_json["categories"].append({
                    "id": category_id,
                    "name": category_name if category_name else f"unknown_{category_id}",
                    "supercategory": ""
                })
                categories_set.add(category_id)

            # Convert bbox from [x, y, w, h] to COCO format [x, y, w, h]
            x, y, width, height = bbox
            coco_bbox = [float(x), float(y), float(width), float(height)]

            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": 1,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": float(width * height),
                "segmentation": detection.get('segmentation', []),
                "iscrowd": 0,
                "score": float(score) if score is not None else 0.0
            }
            coco_json["annotations"].append(annotation)
            annotation_id += 1

        return coco_json

    def publish_failure_result(self, data, error_message):
        """
        Publish a failure result message

        Args:
            data: Original job data
            error_message: Error message
        """
        result_message = {
            "job_id": data.get('job_id'),
            "status": "failed",
            "job_type": "detection",
            "video_id": data.get('video_id'),
            "frame_id": data.get('frame_id'),
            "frame_number": data.get('frame_number'),
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "model": self.args.model,
            "dataset": self.args.dataset
        }

        # Use the publish_results_message function
        publish_results_message(self.channel, result_message)
        self.logger.info(f"Published failure result for job {data.get('job_id')}")

    def upload_frame(self, output_path, s3_frame_location):
        """
        Uploads the image with detections to S3

        Args:
            output_path: Path to the output image
            s3_frame_location: S3 frame location

        Returns:
            S3 path to the uploaded image
        """
        # Extract queue name from class attribute - convert to simple name (detection, ppe, etc.)
        queue_name = self.args.dataset

        # Parse the original S3 path to get the bucket and key
        # Example: s3://bucket-name/path/to/images/file.jpg
        # We want to preserve the bucket name and base structure

        # Extract the base path without /images component
        if "/images/" in s3_frame_location:
            base_path = s3_frame_location.split("/images/")[0]
        else:
            # If no /images/ in path, use the directory part
            base_path = os.path.dirname(s3_frame_location.rstrip('/'))

        # Get the filename from the original path
        filename = os.path.basename(s3_frame_location)

        # Create the new path with queue_name/overlay structure
        s3_overlay_path = f"{base_path}/{queue_name}/overlay/{filename}"

        # Upload the image to S3 - pass empty string as folder to prevent adding /images/
        s3_result_path = upload_frame_to_s3(output_path, s3_overlay_path, folder="")

        return s3_result_path

    def upload_coco_json(self, coco_json, s3_frame_path):
        """
        Uploads the COCO JSON to S3

        Args:
            coco_json: COCO JSON dictionary
            s3_frame_path: Original S3 frame path

        Returns:
            S3 path to the uploaded JSON
        """
        # Create a temporary file for the JSON

        # Explicitly open file in text mode with 'w' instead of default binary mode
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(coco_json, temp_file, indent=2)

        try:
            # Extract queue name from class attribute - be consistent with upload_frame method
            queue_name = self.args.dataset

            # Parse the original S3 path

            bucket, key = parse_s3_path(s3_frame_path)

            # Get base path without /images/ component
            if "/images/" in key:
                base_path = key.split("/images/")[0]
            else:
                # If no /images/ in path, use the directory part
                base_path = os.path.dirname(key)

            # Get the filename from the original path and replace image extension with .json
            filename = os.path.basename(key)
            json_filename = os.path.splitext(filename)[0] + ".json"

            # Create the new path with queue_name/coco_json structure
            s3_key = f"{base_path}/{queue_name}/coco_json/{json_filename}"
            s3_coco_json_path = f"s3://{bucket}/{s3_key}"

            # Use a thread-local S3Transfer to upload
            s3_transfer = get_thread_local_s3_transfer()
            s3_transfer.upload_file(temp_path, bucket, s3_key)

            return s3_coco_json_path
        finally:
            # Remove the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def print_summary(self):
        """
        Print summary of processed images
        """
        if self.total_images > 0:
            average_time = self.total_time / self.total_images
            self.logger.info(f"{self.args.device} | {self.args.model} | Total images processed: {self.total_images}")
            self.logger.info(f"{self.args.device} | {self.args.model} | Total time taken: {self.total_time:.2f} seconds")
            self.logger.info(f"{self.args.device} | {self.args.model} | Average time per image: {average_time:.2f} seconds")
        else:
            self.logger.info(f"{self.args.device} | {self.args.model} | No images were processed.")
