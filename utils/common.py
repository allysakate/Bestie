import os
import cv2
import logging
from datetime import datetime
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import draw_segmentation_masks

from utils import (
    get_instance_segmentation,
)

categories = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

colors = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
    (255, 255, 255),
]


def initiate_logging(log_name: str):
    """Create log file

    Arguments:
        log_name (str):   name of log file

    Returns:
        logging (Logging): logging object

    """
    log_dir = "logs/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = os.path.join(log_dir, f"{log_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)
    now = datetime.now()
    logging.info(f"START: {now}")
    return logging


class ModelConfig(object):
    def __init__(self, config_data):
        self.__dict__.update(config_data)


def get_config(yml_file: str):
    """Get data from config file

    Arguments:
        yml_file (str): config file path

    Returns:
        cfg_dict (Dict[str, object]):   config dict
        model_cfg (ModelConfig):        config class for model parameters

    """
    with open(yml_file, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    model_dict = cfg["main"]
    model_dict.update(cfg["model"])
    model_dict.update(cfg["val"])
    task = cfg["main"]["task"]
    if task == "train":
        model_dict.update(cfg["train"])
        if model_dict["lr"]:
            model_dict["lr"] = float(model_dict["lr"])
    model_cfg = ModelConfig(model_dict)
    return model_cfg


def load_image(img_path):
    img = Image.open(img_path).convert("RGB")

    ori_w, ori_h = img.size

    new_h = (ori_h + 31) // 32 * 32
    new_w = (ori_w + 31) // 32 * 32

    img = img.resize((new_w, new_h), Image.BILINEAR)
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0)
    return img, (ori_h, ori_w)


def get_instance_map(out, target_size, device, model_cfg):
    """
    post-processing (output -> instance map).

    Arguments:
        out: A Dict type for network outputs.
            out['seg']: output semantic segmentation logits
            out['center']: output center map
            out['offset']: output offset map
        cls_label: A Tensor of shape [B, C]. one-hot image-level label.
        target_size: A list for target size (H, W)
        device: output device.
        args: arguments

    Returns:
        seg_map: A numpy array of shape [H, W]. output semantic segmentation map.
        pred_label: A numpy array of shape [H, W]. class-label for output instance mask.
        pred_mask: A numpy array of shape [H, W]. pixel-wise mask for output instance mask.
        pred_score: A numpy array of shape [H, W]. confidence score for output instance mask.
    """
    pred_label, pred_mask, pred_score = [], [], []  # pred_seg
    seg_prob = torch.softmax(out["seg"].detach(), 1)  # B, C+1, H, W
    center_map = out["center"].detach()
    offset_map = out["offset"][0].detach()

    seg_prob = seg_prob[0]
    center_map = center_map[0]

    out_size = seg_prob.shape[1:]
    offset_map[0, :, :] = offset_map[0, :, :] * (target_size[0] / out_size[0])
    offset_map[1, :, :] = offset_map[1, :, :] * (target_size[1] / out_size[1])

    seg_map = torch.argmax(seg_prob, 0)
    valid_cls = torch.unique(seg_map).cpu().numpy() - 1  # -1 for removing bg-class

    for cls in valid_cls:
        if cls < 0:
            continue
        center_map_cls = center_map[cls]  # [H, W]
        fg_cls = (seg_map == (cls + 1)).bool()  # [H, W]

        fg_cls = fg_cls.cpu().numpy().astype(np.uint8)
        n_contours, contours, stats, _ = cv2.connectedComponentsWithStats(
            fg_cls, connectivity=8
        )

        for k in range(1, n_contours):
            size = stats[k, cv2.CC_STAT_AREA]

            if size < model_cfg.minimum_mask_size:
                continue

            contour_mask = contours == k
            contour_mask = torch.from_numpy(contour_mask).to(device)  # [H, W]

            center_map_cls_roi = center_map_cls * contour_mask  # get roi in center_map

            ins_map = get_instance_segmentation(
                contour_mask[None, ...],
                center_map_cls_roi[None, None, ...],
                offset_map[None, ...],
                threshold=model_cfg.val_thresh,
                nms_kernel=model_cfg.val_kernel,
                beta=model_cfg.beta,
                ignore=model_cfg.val_ignore,
            )

            # ins_map : [1, H, W]
            # seg_prob : [21, H, W]
            ins_map = ins_map.squeeze(0)
            n_ins = ins_map.max()

            for id in range(1, n_ins + 1):
                mask = ins_map == id  # [H, W]

                if mask.sum() > 0:
                    index = torch.where(mask)

                    center_idx = center_map_cls_roi[index].argmax()
                    seg_score = seg_prob[cls + 1][index].mean().item()

                    cy, cx = index[0][center_idx], index[1][center_idx]
                    center_score = center_map_cls_roi[cy, cx].item()

                    if center_score >= 1:  # clustered center conf = seg_score
                        center_score = seg_score

                    pred_label.append(cls)
                    pred_mask.append(mask.cpu().numpy())
                    pred_score.append(center_score * seg_score)

    if len(pred_label) == 0:
        pred_label.append(0)
        pred_mask.append(np.zeros(target_size, dtype=np.bool))
        pred_score.append(0)

    pred_label = np.stack(pred_label, 0)
    pred_mask = np.stack(pred_mask, 0)
    pred_score = np.stack(pred_score, 0)

    score_mask = pred_score >= model_cfg.refine_thresh
    pred_label = pred_label[score_mask]
    pred_mask = pred_mask[score_mask]
    pred_score = pred_score[score_mask]

    return seg_map.cpu().numpy(), pred_label, pred_mask, pred_score


def tensor_to_cv2(image_tensor: torch.Tensor):
    """Converts tensor to cv2 image

    Arguments:
        image_tensor (torch.Tensor):image tensor

    Return:
        (np.ndarray):  cv2 image
    """
    image_segm = image_tensor.detach()
    image_segm = TF.to_pil_image(image_segm)
    return cv2.cvtColor(np.array(image_segm), cv2.COLOR_RGB2BGR)


def draw_segm(image_tensor, masks):
    # for mask, color in zip(masks, colors_):
    #     print(mask.shape, color)
    #     img_to_draw[:, mask] = color[:, None]

    # out = image * (1 - alpha) + img_to_draw * alpha
    masks_tensor = torch.from_numpy(masks)
    # bbox = masks_to_boxes(masks_tensor).tolist()
    # print(bbox, len(bbox))
    # bbox = list(map(int, masks_to_boxes(mask_torch)[0]))
    image_tensor = draw_segmentation_masks(
        image_tensor, masks_tensor, alpha=0.8, colors=(0, 255, 0)
    )
    return tensor_to_cv2(image_tensor)
