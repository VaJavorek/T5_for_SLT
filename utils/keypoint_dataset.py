import json
import os
import random
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset

from .normalization import (local_keypoint_normalization, global_keypoint_normalization,
                            yasl_keypoint_normalization, yasl_keypoint_normalization2)

from .augmentations import all_same, get_bbox, use_augmentation, get_rotation_matrix, get_shear_matrix, \
    get_perspective_matrix, apply_transform


def get_keypoints(json_data, data_key='cropped_keypoints', missing_values=0):
    if missing_values is None:
        missing_values = 0

    right_hand_landmarks = []
    left_hand_landmarks = []
    face_landmarks = []
    pose_landmarks = []

    keypoints = json_data[data_key]
    for frame_id in range(len(keypoints)):
        if len(keypoints[frame_id]['pose_landmarks']) == 0:
            _kp = np.zeros((24, 2)) + missing_values
            pose_landmarks.append(_kp)
        else:
            pose_landmarks.append(np.array(keypoints[frame_id]['pose_landmarks']))

        if len(keypoints[frame_id]['right_hand_landmarks']) == 0:
            _kp = np.zeros((21, 2)) + missing_values
            right_hand_landmarks.append(_kp)
        else:
            right_hand_landmarks.append(np.array(keypoints[frame_id]['right_hand_landmarks']))

        if len(keypoints[frame_id]['left_hand_landmarks']) == 0:
            _kp = np.zeros((21, 2)) + missing_values
            left_hand_landmarks.append(_kp)
        else:
            left_hand_landmarks.append(np.array(keypoints[frame_id]['left_hand_landmarks']))

        if len(keypoints[frame_id]['face_landmarks']) == 0:
            _kp = np.zeros((20, 2)) + missing_values
            face_landmarks.append(_kp)
        else:
            face_landmarks.append(np.array(keypoints[frame_id]['face_landmarks']))

    pose_landmarks = np.array(pose_landmarks)#[:, :25]
    return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks


def find_zero_sequences(sequence: list):
    # find starts and ends of zero sub-sequences
    sequence = np.array(sequence)
    sequence = (sequence > 0).astype(int)
    diff = np.diff(np.r_[1, sequence, 1])  # Pad with 1s to detect zero sequence edges
    starts = np.where(diff == -1)[0]  # Start of zero sequences
    ends = np.where(diff == 1)[0]     # End of zero sequences (adjust index)

    return list(zip(starts, ends))


def interpolate_keypoints(keypoints: dict, max_distance: int, round_digits: int = 3):
    # get lengths
    keypoints_lengths = {name: [] for name in keypoints}
    for name, kp in keypoints.items():
        for frame_keypoints in kp:
            length = 0 if all_same(frame_keypoints) else 1
            keypoints_lengths[name].append(length)

    # get sequences
    for name, kp in keypoints_lengths.items():
        sequences = find_zero_sequences(kp)
        for s, e in sequences:
            if (s - 1) <= 0 or e >= len(keypoints[name]):
                continue
            l = e - s
            if l > max_distance:
                continue
            start_keypoints = keypoints[name][s - 1]
            end_keypoints = keypoints[name][e]
            interpolated_keypoints = np.linspace(start_keypoints, end_keypoints, (e - s)+2)[1:-1]
            keypoints[name][s:e] = np.round(interpolated_keypoints, round_digits) # round to same precision as keypoints from json (keypoints are rounded after detection by mediapipe)
    return keypoints


def get_json_files(json_dir):
    json_files = [os.path.join(json_dir, json_file) for json_file in os.listdir(json_dir) if
                  json_file.endswith('.json')]
    return json_files


class KeypointDatasetJSON(Dataset):
    def __init__(
            self,
            json_folder: str,
            clip_to_video: dict = None,
            kp_normalization: tuple = (),
            kp_normalization_method="sign_space",
            data_key: str = "cropped_keypoints",
            missing_values: int = None,
            augmentation_configs: list = [],
            augmentation_per_frame: bool = False,
            interpolate: int = -1,
            load_from_raw=True,
    ):
        """
        Args:
            json_folder: Folder containing raw keypoints in json files or data_key folders containing json files (see load_from_raw).
            clip_to_video: A mapping from clip names to video names.
                           If None each json file will be considered as separate clip.
            kp_normalization: Order and type of normalization tha will be used for individual keypoint groups.
                              For example: ("global-pose_landmarks", "local-face_landmarks")
                                - global normalization for pose and local for face
            kp_normalization_method: What method to use for keypoint normalization:
                    - "" - no normalization - if kp_normalization empty keypoints will be sorted in
                                              order: (pose, right, left, face), else in order kp_normalization
                    - "sign_space" - normalize according to sign space to [-1, 1]
                    - "yasl" - normalization in [0, 1] range across all clip frames
                    - "yasl2" - normalization in [0, 1] range in each frames
            data_key: What data to select from json file
            (cropped_keypoints - keypoints in cropped clip, keypoints - keypoints in original clip)
            missing_values: What value to use for missing values.
            augmentation_configs: List of augmentation configurations.
            augmentation_per_frame: If True, apply augmentation to each frame separately,
                                    else all frames in clip will be augmented in same way.
            interpolate: linear interpolation of keypoints if the missign sequenc is =< than interpolate value
            load_from_raw:  If True, load data from raw json files in json_folder.
                            If False, load data from folder named by data_key in the root directory json_folder.
        """
        if load_from_raw in ["False", "false", False]:
            json_list = get_json_files(os.path.join(json_folder, data_key))
        else:
            json_list = get_json_files(json_folder)
        self.video_to_files = {}
        for idx, path in enumerate(json_list):
            name = os.path.basename(path)
            name_split = name.split(".")[:-1]
            clip_name = ".".join(name_split)

            if clip_to_video is None:
                video_name = clip_name
            else:
                video_name = clip_to_video[clip_name]

            if video_name in self.video_to_files:
                self.video_to_files[video_name].append(path)
            else:
                self.video_to_files[video_name] = [path]
        self.video_names = list(self.video_to_files.keys())
        self.video_name_to_idx = {name: idx for idx, name in enumerate(self.video_to_files)}

        # define keypoint indices for normalization
        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
            93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
            282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
        ]
        self.kp_normalization = kp_normalization
        self.data_key = data_key
        self.missing_values = missing_values
        self.interpolate = interpolate

        # select normalization method
        normalization_methods = {
            "": self._no_normalization,
            "sign_space": self._sign_space_normalization,
            "yasl": self._yasl_normalization,
            "yasl2": self._yasl2_normalization
        }

        if kp_normalization_method not in normalization_methods:
            raise ValueError(f"Unsupported normalization method: {kp_normalization_method}")
        if kp_normalization_method and not kp_normalization:
            raise ValueError("kp_normalization must be provided when using kp_normalization_method")

        self.kp_normalization_method = normalization_methods[kp_normalization_method]
        self.kp_augmentations = KeypointAugmentations(augmentation_configs, augmentation_per_frame) if augmentation_configs else None

    def __len__(self):
        return len(self.video_to_files)

    def load_keypoints(self, file_path):
        """load and prepare keypoints from the json file"""
        with open(file_path, 'r') as file:
            keypoints_meta = json.load(file)
        keypoints = get_keypoints(keypoints_meta, data_key=self.data_key, missing_values=self.missing_values)
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = keypoints
        joints = {
            'face_landmarks': np.array(face_landmarks),
            'left_hand_landmarks': np.array(left_hand_landmarks),
            'right_hand_landmarks': np.array(right_hand_landmarks),
            'pose_landmarks': np.array(pose_landmarks)
        }
        return joints

    def _no_normalization(self, raw_keypoints):
        keypoints_order = ["pose_landmarks", "right_hand_landmarks", "left_hand_landmarks", "face_landmarks"]
        if self.kp_normalization:
            keypoints_order = [kp_name.split("-")[-1] for kp_name in self.kp_normalization]

        data = [raw_keypoints[kp_name] for kp_name in keypoints_order]
        data = np.concatenate(data, axis=1)
        data = data.reshape(data.shape[0], -1)
        return data

    def _sign_space_normalization(self, raw_keypoints):
        local_landmarks = {}
        global_landmarks = {}

        for idx, landmarks in enumerate(self.kp_normalization):
            prefix, landmarks = landmarks.split("-")
            if prefix == "local":
                local_landmarks[idx] = landmarks
            elif prefix == "global":
                global_landmarks[idx] = landmarks

        # local normalization
        for idx, landmarks in local_landmarks.items():
            normalized_keypoints = local_keypoint_normalization(raw_keypoints, landmarks, padding=0.2)
            local_landmarks[idx] = normalized_keypoints

        # global normalization
        additional_landmarks = list(global_landmarks.values())
        if "pose_landmarks" in additional_landmarks:
            additional_landmarks.remove("pose_landmarks")

        keypoints, additional_keypoints = global_keypoint_normalization(
            raw_keypoints,
            "pose_landmarks",
            additional_landmarks,
            l_shoulder_idx = 10,
            r_shoulder_idx = 11
        )

        for k, landmark in global_landmarks.items():
            if landmark == "pose_landmarks":
                global_landmarks[k] = keypoints
            else:
                global_landmarks[k] = additional_keypoints[landmark]

        all_landmarks = {**local_landmarks, **global_landmarks}
        data = []
        for idx in range(len(self.kp_normalization)):
            data.append(all_landmarks[idx])

        if self.missing_values is not None:
            for didx, _data in enumerate(data):
                for fidx in range(len(_data)):
                    if not all_same(_data[fidx]):
                        continue
                    data[didx][fidx] = np.zeros_like(_data[fidx]) + self.missing_values

        data = np.concatenate(data, axis=1)
        data = data.reshape(data.shape[0], -1)
        return data

    def _yasl_normalization(self, raw_keypoints):
        data = []
        for idx, landmarks in enumerate(self.kp_normalization):
            prefix, landmarks = landmarks.split("-")
            data.append(raw_keypoints[landmarks])

        # prepare for missing value replacement
        if self.missing_values is not None:
            landmark_cum_indexes = [0, *np.cumsum([len(i[0]) for i in data])]
            landmark_empty_indexes = []
            for didx, _data in enumerate(data):
                landmark_empty_indexes.append([])
                for fidx in range(len(_data)):
                    if not all_same(_data[fidx]):
                        continue
                    landmark_empty_indexes[didx].append(fidx)

        # normalize
        data = np.concatenate(data, axis=1)
        data = yasl_keypoint_normalization(data)

        # replace missing values
        if self.missing_values is not None:
            for didx in range(len(landmark_empty_indexes)):
                for fidx in landmark_empty_indexes[didx]:
                    sidx = landmark_cum_indexes[didx]
                    eidx = landmark_cum_indexes[didx + 1]
                    data[fidx][sidx:eidx] = np.zeros_like(data[fidx][sidx:eidx]) + self.missing_values


        data = data.reshape(data.shape[0], -1)
        return data

    def _yasl2_normalization(self, raw_keypoints):
        data = []
        for idx, landmarks in enumerate(self.kp_normalization):
            prefix, landmarks = landmarks.split("-")
            data.append(raw_keypoints[landmarks])

        # prepare for missing value replacement
        if self.missing_values is not None:
            landmark_cum_indexes = [0, *np.cumsum([len(i[0]) for i in data])]
            landmark_empty_indexes = []
            for didx, _data in enumerate(data):
                landmark_empty_indexes.append([])
                for fidx in range(len(_data)):
                    if not all_same(_data[fidx]):
                        continue
                    landmark_empty_indexes[didx].append(fidx)

        # normalize
        data = np.concatenate(data, axis=1)
        data = yasl_keypoint_normalization2(data)

        # replace missing values
        if self.missing_values is not None:
            for didx in range(len(landmark_empty_indexes)):
                for fidx in landmark_empty_indexes[didx]:
                    sidx = landmark_cum_indexes[didx]
                    eidx = landmark_cum_indexes[didx+1]
                    data[fidx][sidx:eidx] = np.zeros_like(data[fidx][sidx:eidx]) + self.missing_values

        data = data.reshape(data.shape[0], -1)
        return data

    def get_clip_data(self, clip_name: str) -> np.ndarray:
        """get clip data by its name"""
        idx = self.video_name_to_idx[clip_name]
        clip_data = self[idx]
        return clip_data[0]["data"]

    def __getitem__(self, idx: int) -> list:
        video_name = self.video_names[idx]
        clip_paths = self.video_to_files[video_name]

        output_data = []
        for clip_path in clip_paths:
            name = os.path.basename(clip_path)
            name_split = name.split(".")
            clip_name = ".".join(name_split[:-1])

            keypoints = self.load_keypoints(clip_path)
            keypoints = interpolate_keypoints(keypoints, self.interpolate) if self.interpolate > 0 else keypoints
            keypoints = self.kp_augmentations(keypoints) if self.kp_augmentations is not None else keypoints
            clip_data = self.kp_normalization_method(keypoints)

            clip_data = {"data": clip_data, "video_name": video_name, "clip_name": clip_name}
            output_data.append(clip_data)

        return output_data


class KeypointAugmentations:
    def __init__(self, augmentation_configs, augmentation_per_frame=False):
        self.augmentation_configs = augmentation_configs
        self.augmentation_per_frame = augmentation_per_frame

        self.augmentation_methods = {
            "rotate": self._rotate_augmentation,
            "shear": self._shear_augmentation,
            "perspective": self._perspective_augmentation,
            "rotate_hand": self._rotate_hand_augmentation,
            "noise": self._noise_augmentation
        }

        self._augmentation_config_check(augmentation_configs)

    def _augmentation_config_check(self, config):
        # check augmentation parameters
        for augmentation in config:
            assert "name" in augmentation and "p" in augmentation, \
                f"All augmentation configs must have 'name' and 'p' values specified. {augmentation}"

        # check augmentation names
        for augmentation in config:
            assert augmentation["name"] in self.augmentation_methods, \
                (f"Invalid augmentation name: {augmentation['name']}. "
                 f"Valid augmentations: {list(self.augmentation_methods.keys())}")

    @staticmethod
    def _rotate_augmentation(keypoints: dict, angle: tuple, return_transform=False, transform_matrix=None):
        if transform_matrix is not None:
            rotation_matrix = transform_matrix
        else:
            angle = random.uniform(*angle)
            x0, y0, x1, y1 = get_bbox(keypoints["pose_landmarks"])
            h, w = y1 - y0, x1 - x0
            center = x0 + w / 2, y0 + h / 2
            rotation_matrix = get_rotation_matrix(angle, center)

        for name, kp in keypoints.items():
            if all_same(kp):
                continue
            keypoints[name] = apply_transform(kp, rotation_matrix)

        if return_transform:
            return keypoints, rotation_matrix
        return keypoints

    @staticmethod
    def _shear_augmentation(keypoints: dict, angle_x: tuple, angle_y: tuple, return_transform=False, transform_matrix=None):
        if transform_matrix is not None:
            shear_matrix = transform_matrix
        else:
            angle_x = random.uniform(*angle_x)
            angle_y = random.uniform(*angle_y)
            shear_matrix = get_shear_matrix(angle_x, angle_y)

        for name, kp in keypoints.items():
            if all_same(kp):
                continue
            keypoints[name] = apply_transform(kp, shear_matrix)

        if return_transform:
            return keypoints, shear_matrix
        return keypoints

    @staticmethod
    def _perspective_augmentation(keypoints: dict, portion: tuple, reference_size: int = 512, return_transform=False, transform_matrix=None):
        if transform_matrix is not None:
            perspective_matrix = transform_matrix
        else:
            portion = random.uniform(*portion)
            perspective_matrix = get_perspective_matrix(portion, reference_size)

        for name, kp in keypoints.items():
            if all_same(kp):
                continue
            keypoints[name] = apply_transform(kp, perspective_matrix)

        if return_transform:
            return keypoints, perspective_matrix
        return keypoints

    @staticmethod
    def _rotate_hand_augmentation(keypoints: dict, angle: tuple, rotation_center: str, return_transform=False, transform_matrix=None):
        LEFT_HAND_IDX = [11, 13, 15, 17, 19, 21]
        RIGHT_HAND_IDX = [12, 14, 16, 18, 20, 22]
        rotate_position = {
            "shoulder": 0,
            "elbow": 1,
            "wrist": 2,
        }

        # get angles and rotation position index
        angle_left = random.uniform(*angle)
        angle_right = random.uniform(*angle)
        rot_pos_idx = rotate_position[rotation_center]

        # get indexes for rotation
        left_center_idx = LEFT_HAND_IDX[rot_pos_idx]
        right_center_idx = RIGHT_HAND_IDX[rot_pos_idx]

        left_rotate_idx = LEFT_HAND_IDX[rot_pos_idx:]
        right_rotate_idx = RIGHT_HAND_IDX[rot_pos_idx:]

        # get rotation matrix
        rotation_matrix_left = get_rotation_matrix(angle=angle_left,
                                                   center=keypoints["pose_landmarks"][left_center_idx])
        rotation_matrix_right = get_rotation_matrix(angle=angle_right,
                                                    center=keypoints["pose_landmarks"][right_center_idx])

        # transform
        keypoints["pose_landmarks"][left_rotate_idx] = apply_transform(keypoints["pose_landmarks"][left_rotate_idx],
                                                                       rotation_matrix_left)
        keypoints["pose_landmarks"][right_rotate_idx] = apply_transform(keypoints["pose_landmarks"][right_rotate_idx],
                                                                        rotation_matrix_right)

        if not all_same(keypoints["left_hand_landmarks"]):
            keypoints["left_hand_landmarks"] = apply_transform(keypoints["left_hand_landmarks"], rotation_matrix_left)
        if not all_same(keypoints["right_hand_landmarks"]):
            keypoints["right_hand_landmarks"] = apply_transform(keypoints["right_hand_landmarks"],
                                                                rotation_matrix_right)

        if return_transform:
            return keypoints, (rotation_matrix_left, rotation_matrix_right)

        return keypoints

    @staticmethod
    def _noise_augmentation(keypoints: dict, mean: float = 0, std: float = 1, return_transform=False, transform_matrix=None):
        for name, kp in keypoints.items():
            if all_same(kp):
                continue
            keypoints[name] = kp + np.random.normal(loc=mean, scale=std, size=np.array(kp).shape)

        if return_transform:
            return keypoints, None
        return keypoints

    def __call__(self, keypoints: dict):
        """
        Args:
            keypoints:  {'name': [frames, num_keypoints, 2]}
        """
        _augmentation_configs = deepcopy(self.augmentation_configs)
        _keypoints = deepcopy(keypoints)

        num_frames = len(list(_keypoints.values())[0])

        augmentation_name = [augmentation_config.pop('name') for augmentation_config in _augmentation_configs]
        augmentation_p = [augmentation_config.pop('p') for augmentation_config in _augmentation_configs]
        augmentation_fcn = [self.augmentation_methods[name] for name in augmentation_name]
        augmentation_transform = [None for _ in augmentation_name]
        use_augmentation_clip = [use_augmentation(p) for p in augmentation_p]

        for frame in range(num_frames):
            # get frame keypoints
            frame_keypoints = {}
            for kp_name, landmarks in _keypoints.items():
                frame_keypoints[kp_name] = landmarks[frame]

            # apply augmentation
            for aidx, augmentation_config in enumerate(_augmentation_configs):
                if self.augmentation_per_frame and use_augmentation(augmentation_p[aidx]):
                    frame_keypoints = augmentation_fcn[aidx](keypoints=frame_keypoints, **augmentation_config)

                elif (not self.augmentation_per_frame) and use_augmentation_clip[aidx]:
                    frame_keypoints, _transformation_matrix = augmentation_fcn[aidx](
                        keypoints=frame_keypoints,
                        **augmentation_config,
                        return_transform=True,
                        transform_matrix=augmentation_transform[aidx]
                    )
                    augmentation_transform[aidx] = _transformation_matrix

            # put keypoints back
            for kp_name in _keypoints:
                _keypoints[kp_name][frame] = frame_keypoints[kp_name]

        return _keypoints