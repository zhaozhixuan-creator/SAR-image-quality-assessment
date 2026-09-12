import argparse
import csv
import logging
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PACKAGE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import configs.options as option
import core.evaluation.criteria as criteria
from data import create_dataloader, create_dataset
from models.pipelines import create_pipl
from models.trainscheme import create_scheme
import utils as util


class NVSynthesis:
    """
    NVSynthesis: Top-level wrapper class for training and testing pipeline.
    
    Usage:
        [Train]
            trainer = NVSynthesis(is_train=True)
            trainer.train()
        
        [Test]
            tester = NVSynthesis(is_train=False)
            tester.test()
    """
    def __init__(self, is_train=True, opt=None, test_target_type=None):
        """ initialization

        Args:
            is_train (bool): Whether the mode is training or testing.
            opt (str): Path to the option YAML file.
            test_target_type (str): Optional test target filter. Use "all" to
                evaluate every target in the test split.
        """
        self.is_train = is_train
        self.opt = option.parse(opt, is_train=self.is_train) # load options from YAML file
        self._override_test_target_type(test_target_type)
        self.init_setup() # setup: env, data-related, pipL & model, logger

    def _override_test_target_type(self, test_target_type):
        if self.is_train:
            return
        if not self.opt["setting"].get("test_target_type"):
            self.opt["setting"]["test_target_type"] = "ZIL131"
        if test_target_type is not None:
            self.opt["setting"]["test_target_type"] = test_target_type
    
    def init_setup(self):
        """ Run full environment setup
        """
        self.setup_environment() # random_seed & self.device
        self.setup_dir()         # mkdir
        self.setup_logging()     # self.logger
        self.setup_data()        # dataloader
        self.setup_model()       # pipL -> trainscheme -> model

    def setup_environment(self):        
        """
        Set random seed and CUDA settings.
        """
        seed = self.opt["setting"]["manual_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_dir(self):
        """
        Set up directories for logs, models, and results.
        """
        # Get root dir
        if self.is_train:
            root_dir_name = "experiments_root"
        else:
            root_dir_name = "results_root"
        util.mkdir(self.opt["path"][root_dir_name])

        # Create all other directories
        util.mkdirs(
            (
                path
                for key, path in self.opt["path"].items()
                if not key == root_dir_name
                and "pretrain_model" not in key
                and "pretrain" not in key
            )
        )
        # Set up log symlink
        log_link = Path(__file__).with_name("log")
        if log_link.is_symlink() or log_link.is_file():
            log_link.unlink()
        os.symlink(os.path.join(self.opt["path"][root_dir_name], ".."), log_link)
    
    def setup_logging(self):
        """
        Configure loggers and tensorboard writer.
        """
        # config loggers. Before it, the log will not work
        if self.opt["is_train"]:
            util.setup_logger(
                "base",
                self.opt["path"]["log"],
                "train_" + self.opt["name"],
                level=logging.INFO,
                screen=True,
                tofile=True,
            )
            util.setup_logger(
                "val",
                self.opt["path"]["log"],
                "val_" + self.opt["name"],
                level=logging.INFO,
                screen=True,
                tofile=True,
            )
        else:
            util.setup_logger(
                "base",
                self.opt["path"]["log"],
                "test_" + self.opt["name"] + "_" + self.opt["pretrain"],
                level=logging.INFO,
                screen=True,
                tofile=True,
            )
        self.logger = logging.getLogger("base")
        self.logger.info(option.dict2str(self.opt))

        if self.opt["use_tb_logger"] and "debug" not in self.opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                self.logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            self.tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(self.opt["name"]))
       
    def setup_data(self):
        """
        Create dataloaders for train/val/test sets.
        """
        if self.opt["is_train"]:
            self.train_loader_list, self.val_loader_list = [], []
            self.total_epochs_list = []

            # Train datasets
            for name, dataset_opt in self.opt["datasets"]["train"].items():
                train_set = create_dataset(dataset_opt)

                train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
                total_iters = int(self.opt["setting"]["niter"])
                total_epochs = int(math.ceil(total_iters / train_size))

                train_loader = create_dataloader(train_set, dataset_opt)
                
                self.train_loader_list.append(train_loader)
                self.total_epochs_list.append(total_epochs)
                

                self.logger.info(
                    "Train Dataset [{:s}]".format(name)
                )
                self.logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                self.logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )

            # Validation datasets
            for name, dataset_opt in self.opt["datasets"]["val"].items():
                val_set = create_dataset(dataset_opt)
                val_loader = create_dataloader(val_set, dataset_opt)
                self.val_loader_list.append(val_loader)
                self.logger.info(
                    "Val Dataset [{:s}]".format(name)
                )
                self.logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            self.test_loader_list = []
            self.test_dataset_opt = []

            # Test datasets
            for name, dataset_opt in self.opt["datasets"]["test"].items():
                test_set = create_dataset(dataset_opt)
                dataset_opt["index_start"] = test_set.index_start
                dataset_opt["target_types"] = list(getattr(test_set, "index_map", {}).keys())
                test_loader = create_dataloader(test_set, dataset_opt)
                self.test_loader_list.append(test_loader)
                self.test_dataset_opt.append(dataset_opt)
                self.logger.info(
                    "Test Dataset [{:s}]".format(name)
                )
                self.logger.info(
                    "Number of test images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(test_set)
                    )
                )

    def setup_model(self):
        """
        Initialize training scheme and model pipeline.
        """
        self.scheme = create_scheme(self.opt)
        self.pipL = create_pipl(self.opt, device=self.scheme.device)
        if "debug" not in self.opt["name"]:
            self.resume_training()
        self.pipL.set_scheme(self.scheme)
    
    def resume_training(self):
        """
        Resume training from checkpoint if specified.
        """
        if self.opt["pretrain"]:
            self.logger.info(
                "Resuming training from: {}".format(self.opt["path"]["pretrain_model_pth"])
            )
  
            self.scheme.resume_training(self.opt["path"]["pretrain_model_pth"])  # handle optimizers and schedulers
            # resume_state["epoch"] = 0
            # resume_state["iter"] = 0
            self.current_step = 0
            self.start_epoch = 0
        else:
            self.current_step = 0
            self.start_epoch = 0
        if self.opt["is_train"]:
            self.logger.info(
                "Start training from epoch: {:d}, iter: {:d}".format(self.start_epoch, self.current_step)
            )

    def train(self):
        """ Training """
        current_step = 0
        start_epoch = 0

        best_psnr = 0.0
        best_iter = 0

        # Training loop: different dataset

        for (dataset_opt, train_loader, total_epochs, val_loader) in zip(
                            self.opt["datasets"]["train"].values(),
                            self.train_loader_list,
                            self.total_epochs_list,
                            self.val_loader_list):
            
            # Training loop: epoch
           
            for epoch in range(start_epoch, total_epochs + 1):

                # Training loop: iter
                for train_data in train_loader:
                    current_step += 1
                    if current_step > self.opt["setting"]["niter"]:
                        break

                    # add auxilary info to the data
                    train_data.update(dataset_opt)
                    train_data.update({"epoch": epoch, "current_step": current_step})
                    self.pipL.Train_Pipeline(**train_data)

                    # validation
                    if current_step % self.opt["setting"]["val_freq"] == 0:
                        self.validate(val_loader, epoch, current_step, best_psnr, best_iter)

                    # save
                    if current_step % self.opt["logger"]["save_checkpoint_freq"] == 0:
                        self.save_model_and_state(epoch, current_step)
        
        # Training loop: END

        self.logger.info("Saving the final model.")
        self.scheme.save("latest")
        self.logger.info("End of Predictor and Corrector training.")


    def validate(self, val_loader, epoch, current_step, best_psnr, best_iter):
        """ Validation with evaluation and logging """
        avg_psnr = 0.0
        for val_data in val_loader:
            # Generate one validation sample; image saving is handled by test().
            self.pipL.record = [[
                                    val_data["img"][:, 0], 
                                    val_data["azimuth_angle"][:, 0].unsqueeze(1),
                                    val_data["incidence_angle"][:, 0].unsqueeze(1)
                                ]]
            output_dict = self.pipL.Test_Pipeline(**val_data)
            pred_LQ = output_dict["z"]
            GT_LQ = val_data["img"][:, 1]
            output = util.tensor2img(pred_LQ.squeeze())
            gt_img = util.tensor2img(GT_LQ.squeeze())
            avg_psnr += criteria.calculate_psnr(output, gt_img)
            break  # ATTENTION!!! Single validation step for simplicity

        avg_psnr /= len(val_loader) # ATTENTION!!!

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_iter = current_step

        self.logger.info(f"# Validation # PSNR: {avg_psnr:.6f}, Best PSNR: {best_psnr:.6f} | Iter: {best_iter}")
        logging.getLogger("val").info(f"<epoch:{epoch:3d}, iter:{current_step:8,d}, psnr: {avg_psnr:.6f}")

    @staticmethod
    def _sanitize_filename(value):
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
        return value.strip("._") or "sample"

    @staticmethod
    def _unique_path(path):
        if not os.path.exists(path):
            return path

        root, ext = os.path.splitext(path)
        suffix = 1
        while True:
            candidate = f"{root}_{suffix:03d}{ext}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    @staticmethod
    def _angle_to_deg(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        return float(value) * 180.0 / math.pi

    @staticmethod
    def _circular_delta_deg(angle_a, angle_b):
        diff = abs(float(angle_a) - float(angle_b)) % 360.0
        return min(diff, 360.0 - diff)

    @staticmethod
    def _target_type_from_data(test_data):
        img_name = test_data.get("img_name", "unknown_target")
        if isinstance(img_name, (list, tuple)):
            return str(img_name[0])
        return str(img_name)

    @staticmethod
    def _normalize_target_type(target_type):
        return re.sub(r"[^A-Za-z0-9]+", "", str(target_type)).upper()

    def _test_target_filter(self):
        target_type = self.opt["setting"].get("test_target_type", "ZIL131")
        if target_type is None or str(target_type).lower() == "all":
            return None
        return str(target_type)

    def _should_test_target(self, target_type):
        selected_target = self._test_target_filter()
        if selected_target is None:
            return True
        return self._normalize_target_type(target_type) == self._normalize_target_type(selected_target)

    def _target_filter_label(self):
        selected_target = self._test_target_filter()
        return "all" if selected_target is None else selected_target

    @staticmethod
    def _tensor_to_uint8_array(tensor):
        tensor = tensor.detach().float().cpu().squeeze()
        if tensor.dim() == 2:
            img = tensor
        elif tensor.dim() == 3 and tensor.size(0) == 1:
            img = tensor[0]
        elif tensor.dim() == 3 and tensor.size(0) == 3:
            img = tensor.permute(1, 2, 0)
        elif tensor.dim() == 3:
            img = torch.sqrt(torch.mean(torch.pow(tensor, 2), dim=0))
        else:
            raise TypeError(f"Unsupported image tensor dimension: {tensor.dim()}")

        img = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return img.numpy()

    def _extract_angle_metrics(self, test_data):
        input_azimuth = self._angle_to_deg(test_data["azimuth_angle"][0, 0])
        target_azimuth = self._angle_to_deg(test_data["azimuth_angle"][0, 1])
        input_incidence = self._angle_to_deg(test_data["incidence_angle"][0, 0])
        target_incidence = self._angle_to_deg(test_data["incidence_angle"][0, 1])

        input_elevation = 90.0 - input_incidence
        target_elevation = 90.0 - target_incidence

        return {
            "input_azimuth": input_azimuth,
            "input_elevation": input_elevation,
            "target_azimuth": target_azimuth,
            "target_elevation": target_elevation,
            "delta_azimuth": self._circular_delta_deg(input_azimuth, target_azimuth),
            "delta_elevation": abs(input_elevation - target_elevation),
        }

    def _select_min_guidance_result(self, pred_lq):
        min_w_idx = int(torch.argmin(self.pipL.w).item())
        if min_w_idx >= pred_lq.shape[0]:
            raise IndexError(
                f"Minimum guidance index {min_w_idx} exceeds output batch size {pred_lq.shape[0]}"
            )
        return min_w_idx, float(self.pipL.w[min_w_idx].item()), pred_lq[min_w_idx].detach().cpu()

    def _sync_if_cuda(self):
        if torch.cuda.is_available() and self.scheme.device.type == "cuda":
            torch.cuda.synchronize(self.scheme.device)

    def _add_panel_label(self, image, label):
        image = image.convert("RGB")
        title_height = 18
        panel = Image.new("RGB", (image.width, image.height + title_height), "white")
        panel.paste(image, (0, title_height))

        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = max(0, (image.width - text_width) // 2)
        draw.text((text_x, 3), label, fill=(0, 0, 0), font=font)
        return panel

    def _save_comparison_image(self, sample_id, test_data, generated_result):
        output_dir = os.path.join(self.opt["path"]["results_root"], "comparison_images")
        os.makedirs(output_dir, exist_ok=True)

        panels = [
            ("condition_image", test_data["img"][0, 0]),
            ("ground_truth", test_data["img"][0, 1]),
            ("generated_result", generated_result),
        ]
        labeled_panels = []
        for label, tensor in panels:
            image = Image.fromarray(self._tensor_to_uint8_array(tensor))
            labeled_panels.append(self._add_panel_label(image, label))

        gap = 6
        width = sum(panel.width for panel in labeled_panels) + gap * (len(labeled_panels) - 1)
        height = max(panel.height for panel in labeled_panels)
        canvas = Image.new("RGB", (width, height), "white")

        x_offset = 0
        for panel in labeled_panels:
            canvas.paste(panel, (x_offset, 0))
            x_offset += panel.width + gap

        target_type = self._sanitize_filename(self._target_type_from_data(test_data))
        angles = self._extract_angle_metrics(test_data)
        filename = (
            f"{self._sanitize_filename(sample_id)}_{target_type}"
            f"_az{angles['target_azimuth']:.2f}_el{angles['target_elevation']:.2f}.png"
        )
        save_path = self._unique_path(os.path.join(output_dir, filename))
        canvas.save(save_path)
        return save_path

    def _build_metric_row(
        self,
        sample_id,
        dataset_opt,
        test_data,
        guidance_w,
        generation_time_sec,
        metrics,
        result_image_path,
    ):
        angle_metrics = self._extract_angle_metrics(test_data)
        row = {
            "sample_id": sample_id,
            "experiment_name": self.opt["name"],
            "pretrain": self.opt["pretrain"],
            "dataset_name": dataset_opt["name"],
            "target_type": self._target_type_from_data(test_data),
            "guidance_w": round(float(guidance_w), 6),
            "generation_time_sec": round(float(generation_time_sec), 6),
            "MSE": round(float(metrics["target_region_MSE"]), 10),
            "target_region_MSE": round(float(metrics["target_region_MSE"]), 10),
            "full_image_MSE": round(float(metrics["full_image_MSE"]), 10),
            "target_weighted_MSE": round(float(metrics["target_weighted_MSE"]), 10),
            "target_pixel_ratio": round(float(metrics["target_pixel_ratio"]), 6),
            "target_mse_top_ratio": round(float(metrics["target_mse_top_ratio"]), 6),
            "result_image_path": result_image_path,
        }
        for key, value in angle_metrics.items():
            row[key] = round(float(value), 6)
        return row

    def _write_metrics_summary(self, rows):
        if not rows:
            return None

        csv_path = os.path.join(self.opt["path"]["results_root"], "evaluation_metrics_summary.csv")
        fieldnames = [
            "sample_id",
            "experiment_name",
            "pretrain",
            "dataset_name",
            "target_type",
            "input_azimuth",
            "input_elevation",
            "target_azimuth",
            "target_elevation",
            "delta_azimuth",
            "delta_elevation",
            "generation_time_sec",
            "MSE",
            "target_region_MSE",
            "full_image_MSE",
            "target_weighted_MSE",
            "target_pixel_ratio",
            "target_mse_top_ratio",
            "guidance_w",
            "result_image_path",
        ]
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path
    
    def test(self):
        """ Testing """
        metrics_rows = []
        # testing loop: different dataset
        for (test_loader, dataset_opt) in zip(
                            self.test_loader_list,
                            self.test_dataset_opt):
            total_psnr = 0.0
            total_mse = 0.0
            num_evaluated = 0
            num_skipped = 0
            self.logger.info(f"Test target filter for [{dataset_opt['name']}]: {self._target_filter_label()}")
            selected_target = self._test_target_filter()
            if selected_target is not None:
                available_targets = dataset_opt.get("target_types", [])
                if available_targets and not any(self._should_test_target(t) for t in available_targets):
                    self.logger.warning(
                        f"Target [{selected_target}] is not present in test dataset "
                        f"[{dataset_opt['name']}]. Available targets: {available_targets}"
                    )
            # testing loop: data
            for idx_data, test_data in enumerate(test_loader):
                target_type = self._target_type_from_data(test_data)
                if not self._should_test_target(target_type):
                    num_skipped += 1
                    continue

                if idx_data in dataset_opt["index_start"]:
                    self.pipL.record = [[
                                            test_data["img"][:, 0], 
                                            test_data["azimuth_angle"][:, 0].unsqueeze(1),
                                            test_data["incidence_angle"][:, 0].unsqueeze(1)
                                         ]]
                    self.logger.info(f"Target type: {target_type}")
                    self.logger.info(f"Condition # Azimuth: {test_data['azimuth_angle'][:, 0].item() / math.pi * 180:.6f} | Incidence: {test_data['incidence_angle'][:, 0].item() / math.pi * 180:.6f}")
                    continue
                if (idx_data - 1) in dataset_opt["index_start"] or (idx_data - 2) in dataset_opt["index_start"]:
                    self.pipL.record = [[
                                            test_data["img"][:, 1], 
                                            test_data["azimuth_angle"][:, 1].unsqueeze(1),
                                            test_data["incidence_angle"][:, 1].unsqueeze(1)
                                         ]]
                    self.logger.info(f"Condition # Azimuth: {test_data['azimuth_angle'][:, 1].item() / math.pi * 180:.6f} | Incidence: {test_data['incidence_angle'][:, 1].item() / math.pi * 180:.6f}")
                    continue

                sample_id = f"{dataset_opt['name']}_{idx_data:06d}"
                self._sync_if_cuda()
                start_time = time.perf_counter()
                output_dict = self.pipL.Test_Pipeline(**test_data)
                self._sync_if_cuda()
                generation_time_sec = time.perf_counter() - start_time

                pred_LQ = output_dict["z"]
                min_w_idx, guidance_w, generated_result = self._select_min_guidance_result(pred_LQ)
                gt_lq = test_data["img"][0, 1].detach().cpu()

                result_image_path = self._save_comparison_image(
                    sample_id,
                    test_data,
                    generated_result,
                )
                target_mse_top_ratio = float(self.opt["setting"].get("target_mse_top_ratio", 0.2))
                target_mse_weight_power = float(self.opt["setting"].get("target_mse_weight_power", 1.0))
                metrics = {
                    "target_region_MSE": criteria.calculate_target_region_mse(
                        generated_result,
                        gt_lq,
                        min_max=(-1, 1),
                        top_ratio=target_mse_top_ratio,
                    ),
                    "full_image_MSE": criteria.calculate_normalized_mse(
                        generated_result,
                        gt_lq,
                        min_max=(-1, 1),
                    ),
                    "target_weighted_MSE": criteria.calculate_weighted_mse(
                        generated_result,
                        gt_lq,
                        min_max=(-1, 1),
                        weight_power=target_mse_weight_power,
                    ),
                    "target_pixel_ratio": criteria.calculate_top_intensity_ratio(
                        gt_lq,
                        min_max=(-1, 1),
                        top_ratio=target_mse_top_ratio,
                    ),
                    "target_mse_top_ratio": target_mse_top_ratio,
                }
                output = self._tensor_to_uint8_array(generated_result)
                gt_img = self._tensor_to_uint8_array(gt_lq)
                psnr = criteria.calculate_psnr(output, gt_img)

                self.pipL.record.append([
                                            generated_result,
                                            test_data["azimuth_angle"][0, 1].detach().cpu().unsqueeze(0).unsqueeze(0),
                                            test_data["incidence_angle"][0, 1].detach().cpu().unsqueeze(0).unsqueeze(0)
                                        ])

                metrics_rows.append(
                    self._build_metric_row(
                        sample_id,
                        dataset_opt,
                        test_data,
                        guidance_w,
                        generation_time_sec,
                        metrics,
                        result_image_path,
                    )
                )
                total_psnr += psnr
                total_mse += metrics["target_region_MSE"]
                num_evaluated += 1

                self.logger.info(f"Target    # Azimuth: {test_data['azimuth_angle'][0, 1].item() / math.pi * 180:.6f} | Incidence: {test_data['incidence_angle'][0, 1].item() / math.pi * 180:.6f}")
                self.logger.info(
                    f"# Sample: {sample_id} # w_idx: {min_w_idx} # w: {guidance_w:.6f} "
                    f"# generation_time_sec: {generation_time_sec:.6f} "
                    f"# MSE(target_region): {metrics['target_region_MSE']:.10f} "
                    f"# full_image_MSE: {metrics['full_image_MSE']:.10f}"
                )

            if num_evaluated > 0:
                avg_psnr = total_psnr / num_evaluated
                avg_mse = total_mse / num_evaluated
                self.logger.info(f"# PSNR: {avg_psnr:.6f} # MSE(target_region): {avg_mse:.10f}")
            else:
                self.logger.info(f"No generated test samples were evaluated for [{dataset_opt['name']}].")
            if num_skipped > 0:
                self.logger.info(f"Skipped {num_skipped} samples outside target filter [{self._target_filter_label()}].")

        # testing loop: END
        metrics_csv_path = self._write_metrics_summary(metrics_rows)
        if metrics_csv_path:
            self.logger.info(f"Evaluation metrics summary saved to: {metrics_csv_path}")
    
    # Save function
    def save_model_and_state(self, epoch, current_step):
        """Save model and training state"""
        self.logger.info("Saving models and training states.")
        self.scheme.save(epoch, current_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument(
        "--test-target-type",
        type=str,
        default=None,
        help="Target type to evaluate in test mode. Use 'all' to evaluate the full test split. Defaults to ZIL131.",
    )
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = NVSynthesis(is_train=True, opt=args.opt)
        trainer.train()
    elif args.mode == 'test':
        tester = NVSynthesis(is_train=False, opt=args.opt, test_target_type=args.test_target_type)
        tester.test()
