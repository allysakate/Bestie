"""
Usage:
1. Modify configs/bestie.yml
2: Run: python test.py

"""
import os
import argparse
from glob import glob
import cv2
import torch
from models import model_factory
from utils.common import (
    get_config,
    load_image,
    initiate_logging,
    get_instance_map,
    draw_segm,
    colors,
    categories,
)
from utils.progbar import ProgressBar
from torchvision.io import read_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BESTIE pytorch implementation")
    parser.add_argument(
        "--cfg", type=str, default="configs/bestie_voc.yaml", help="configuration file"
    )
    args = parser.parse_args()
    # Get config
    cfg_file = args.cfg
    model_cfg = get_config(args.cfg)

    # Initialize logger
    log_name = "test_" + os.path.splitext(os.path.basename(cfg_file))[0]
    logger = initiate_logging(log_name)

    # Sets torch device
    if torch.cuda.is_available():
        device_name = f"cuda:{model_cfg.gpu}"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    # Create save folder
    save_dir = model_cfg.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    model = model_factory(model_cfg)
    model = model.to(device)
    checkpoint = torch.load(model_cfg.checkpoint)
    model.load_state_dict(checkpoint["model"])
    # logger.info(f"model: {model}\n")
    model.eval()

    # Get image paths and limit of images to test
    image_paths = glob(os.path.join(model_cfg.root_dir, "*.jpg"))
    img_set_limit = model_cfg.img_limit
    if len(image_paths) > img_set_limit:
        img_limit = img_set_limit
    else:
        img_limit = len(image_paths)

    try:
        prog_bar = ProgressBar(img_limit, fmt=ProgressBar.FULL)
        img_cnt = 1
        for image_path in image_paths:
            try:
                image_name = os.path.basename(image_path)
                img, orig_size = load_image(image_path)
                target_size = int(orig_size[0]), int(orig_size[1])
                out = model(img.to(device), target_shape=target_size)
                seg_map, pred_label, pred_mask, pred_score = get_instance_map(
                    out, target_size, device, model_cfg
                )
                if pred_score.any():
                    image_tensor = read_image(image_path)
                    cv2_image = draw_segm(image_tensor, pred_mask)
                    idx = 0
                    for label, score in zip(pred_label, pred_score):
                        cv2.putText(
                            cv2_image,
                            f"{categories[label]}: {round(score, 2)}",
                            (0, 0 + (idx * 20)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.5,
                            colors[label],
                            thickness=2,
                        )
                        idx += 1
                    out_path = os.path.join(save_dir, image_name)
                    cv2.imwrite(out_path, cv2_image)
            except ValueError as err:
                logger.error(err)
                logger.error(f"image: {image_name}, size: {img.shape}")

            if img_cnt > img_limit:
                break
            # Update progress bar
            prog_bar.current += 1
            prog_bar()
            img_cnt += 1
        prog_bar.done()
    except Exception as err:
        logger.error(err)
    logger.info("---------------END---------------\n")
