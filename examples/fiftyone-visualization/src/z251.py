import os
import sys

import fiftyone as fo
import open3d as o3d
import yaml
from tqdm import tqdm
from utils import normalize_bbox, quaternion_to_euler

import zod.constants as constants
from zod import ZodFrames
from zod.constants import AnnotationProject, Anonymization


def get_dataset_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
      config_path (str): Path to the configuration YAML file.

    Returns:
      dict: The loaded configuration dictionary.
    """
    # Load config using appropriate library (e.g., yaml)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_directories(config: dict) -> None:
    """
    Creates necessary directories for the dataset based on configuration.

    Args:
      config (dict): The dataset configuration dictionary.
    """
    pcd_files_dir = config["pcd_files_dir"]
    os.makedirs(pcd_files_dir, exist_ok=True)


def filter_zod_frames(zod_frames: ZodFrames, dataset_split: str, config: dict) -> list[str]:
    """
    Filters ZOD frames based on the specified dataset split.

    Args:
      zod_frames (ZodFrames): The ZodFrames object containing all frames.
      dataset_split (str): The desired dataset split ("all", "train", or "val").
      config (dict): The dataset configuration dictionary (for handling invalid splits).

    Returns:
      list: A list of ZOD frame IDs based on the split.
    """
    if dataset_split not in ["all", "train", "val"]:
        print("Invalid dataset_split specified.")
        sys.exit()

    if dataset_split == "all":
        return list(zod_frames.get_all_ids())
    else:
        # constants = get_constants()  # Assuming constants module exists
        return list(
            zod_frames.get_split(constants.TRAIN if dataset_split == "train" else constants.VAL)
        )


def process_zod_frame(zod_frame: ZodFrames, pcd_files_dir: str) -> tuple[str, list, str]:
    # TODO convert to pcd
    """
    Processes a single ZOD frame, extracting relevant data.

    Args:
      zod_frame (ZodFrame): The ZOD frame object.
      pcd_files_dir (str): Path to the directory for storing point cloud files.

    Returns:
      tuple: A tuple containing (core_image_path, annotations, pcd_filename).
    """
    camera_core_frame = zod_frame.info.get_key_camera_frame(Anonymization.BLUR)
    core_image_path = camera_core_frame.filepath
    annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)

    # Handle point cloud creation (similar to original code)
    pcd_filename = f"{pcd_files_dir}/{zod_frame.info.id}.pcd"
    if os.path.exists(pcd_filename):
        return core_image_path, annotations, pcd_filename
        # ... (logic for creating point cloud from core_lidar)
    return core_image_path, annotations, pcd_filename


def convert_annotations(annotations: list) -> tuple[list[fo.Detection], list[fo.Detection]]:
    # TODO split into 2 funcs?
    """
    Converts ZOD annotations into FiftyOne detection format.

    Args:
      annotations (list): A list of ZOD annotations.

    Returns:
      tuple: A tuple containing two lists (detections_3d, detections_2d).
    """
    detections_3d, detections_2d = [], []
    for anno in annotations:
        if anno.box3d is not None:
            # ... (logic for creating 3D detections, similar to original code)
            location = anno.box3d.center
            dimensions = anno.box3d.size
            qw = anno.box3d.orientation[0]
            qx = anno.box3d.orientation[1]
            qy = anno.box3d.orientation[2]
            qz = anno.box3d.orientation[3]
            rotation = quaternion_to_euler(qx, qy, qz, qw)

            detection_3d = fo.Detection(
                dimensions=list(dimensions),
                location=list(location),
                rotation=list(rotation),
                label=anno.name,
            )
            detections_3d.append(detection_3d)

            # ... (logic for creating 2D detections, similar to original code)
            detection_2d = fo.Detection(
                bounding_box=normalize_bbox(anno.box2d.xywh),
                label=anno.name,
            )
            detections_2d.append(detection_2d)
        else:
            pass

    return detections_3d, detections_2d


def create_dataset_samples(zod_frame: ZodFrames, pcd_filename: str, config: dict) -> list:
    """
    Creates FiftyOne samples (image and point cloud) with detections.

    Args:
      zod_frame (ZodFrame): The ZOD frame object.
      pcd_filename (str): Path to the point cloud file.
      config (dict): The dataset configuration dictionary.

    Returns:
      list: A list of FiftyOne Sample objects.
    """
    core_image_path, annotations, _ = process_zod_frame(zod_frame, config["pcd_files_dir"])
    detections_3d, detections_2d = convert_annotations(annotations)

    group = fo.Group()
    samples = [
        fo.Sample(
            filepath=core_image_path,
            group=group.element("image"),
            detections=fo.Detections(detections=detections_2d),
        ),
        fo.Sample(
            filepath=pcd_filename,
            group=group.element("pcd"),
            detections=fo.Detections(detections=detections_3d),
        ),
    ]

    def add_metadata(slice_idx):
        samples[slice_idx]["frame_id"] = zod_frame.metadata.frame_id
        samples[slice_idx]["time_of_day"] = zod_frame.metadata.time_of_day
        samples[slice_idx]["country_code"] = zod_frame.metadata.country_code
        samples[slice_idx]["collection_car"] = zod_frame.metadata.collection_car
        samples[slice_idx]["road_type"] = zod_frame.metadata.road_type
        samples[slice_idx]["road_condition"] = zod_frame.metadata.road_condition
        samples[slice_idx]["num_vehicles"] = zod_frame.metadata.num_vehicles
        samples[slice_idx]["location"] = fo.GeoLocation(
            point=[zod_frame.metadata.longitude, zod_frame.metadata.latitude]
        )

    add_metadata(0)  # add metadata to images
    # add_metadata(1) # add metadata to point clouds (optional)

    return samples


def create_fiftyone_database(config: dict, samples: list) -> None:
    """
    Creates a FiftyOne dataset from ZOD frames with point clouds and annotations.

    Args:
      config (dict): Configuration dictonary.
      samples (list) : List of datas loaded from ZOD.
    """
    dataset = fo.Dataset(name=config["dataset_name"])
    dataset.add_samples(samples)

    # colour by label values by default
    # and change to colour blind friendly colour scheme
    # TODO maybe this app config stuff can be it's own func?
    dataset.app_config.color_scheme = fo.ColorScheme(
        color_by="value",
        color_pool=[
            "#E69F00",
            "#56b4e9",
            "#009e74",
            "#f0e442",
            "#0072b2",
            "#d55e00",
            "#cc79a7",
        ],
    )

    # if mapbox token is provided, add it to the app config
    if config["mapbox_token"]:
        print("Mapbox token found, enabling map plugin.")
        dataset.app_config.plugins["map"] = {"mapboxAccessToken": config["mapbox_token"]}
    else:
        print("Mapbox token not found, map plugin not enabled.")

    dataset.save()

    # keep dataset after session is terminated or not - set in config.yaml
    dataset.persistent = config["dataset_persistent"]


def create_dataset(config_path: str) -> None:
    """
    Creates a FiftyOne dataset from ZOD frames with point clouds and annotations.

    Args:
      config_path (str): Path to the configuration YAML file.
    """
    config = get_dataset_config(config_path)
    create_dataset_directories(config)

    zod_frames = ZodFrames(dataset_root=config["dataset_root"], version=config["dataset_version"])
    zod_frame_list = filter_zod_frames(zod_frames, config["dataset_split"], config)

    samples = []

    for idx in tqdm(zod_frame_list):
        zod_frame = zod_frames[idx]
        pcd_filename = process_zod_frame(zod_frame, config["pcd_files_dir"])[
            2
        ]  # TODO if weareusing only index2 do we need to return the others?
        sample_list = create_dataset_samples(zod_frame, pcd_filename, config)
        samples.extend(sample_list)

    create_fiftyone_database(config=config, samples=samples)


if __name__ == "__main__":
    create_dataset(
        config_path="/home/general_zod/Golam/zod/examples/fiftyone-visualization/src/config.yaml"
    )
