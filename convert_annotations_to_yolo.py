import json

def convert_annotations_to_yolo(json_file_path, output_folder):
    # Load JSON data from file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    video_name = "avd13_cam3_20220302140230_20220302140730.mp4"
    video_info = json_data["item"]["slots"][0]  # single frame
    video_width = video_info["width"]
    video_height = video_info["height"]

    yolo_annotations = {}  # Dict to store YOLO annotations, key is frame number

    for obj in json_data["annotations"]:
        for frame, data in obj["frames"].items():
            bbox = data["bounding_box"]
            x_center = (bbox["x"] + bbox["w"] / 2) / video_width
            y_center = (bbox["y"] + bbox["h"] / 2) / video_height
            width = bbox["w"] / video_width
            height = bbox["h"] / video_height

            class_id = 0  # only one class 'pig'
            annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            if frame not in yolo_annotations:
                yolo_annotations[frame] = [annotation_line]
            else:
                yolo_annotations[frame].append(annotation_line)

    # Writing annotations to text files, one per frame
    for frame, annotations in yolo_annotations.items():
        filename = f"{video_name}_{frame}.txt"
        with open(f"{output_folder}/{filename}", 'w') as file:
            for annotation in annotations:
                file.write(annotation + "\n")



json_file_path = '/cluster/home/muhammmo/pig_dataset/train/vid21/annotations/avd13_cam3_20220302140230_20220302140730.json'
output_folder = '/cluster/home/muhammmo/pig_dataset/vid21_yolo_ann'
convert_annotations_to_yolo(json_file_path, output_folder)
