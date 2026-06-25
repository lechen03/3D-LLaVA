"""Two-stage referring segmentation: predict the object location, then segment.

Stage 1 (grounding): given a description, the model emits [LOC] and a regression
head predicts the object's 3D center (in the post-transform coord frame).
Stage 2 (seg): the predicted center is converted to the nearest superpoint(s) and
fed as the <loc> click to the existing click-based referring-segmentation path,
which predicts the [SEG] mask.

This mirrors llava/eval/model_scanrefer_with_click.py, but replaces the oracle GT
click with the model's own predicted location.
"""
import argparse
import math
import torch
import os
import json
from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.pc_utils import referseg_transform_eval, Compose
import pathlib
import numpy as np
from pointgroup_ops import voxelization_idx
from typing import Sequence, Mapping
from torch_geometric.utils import scatter


# stage 1: predict the object location ([LOC])
GROUNDING_TEMPLATE = (
    "<image>\n Please output the location of the object according to the "
    "following description. \n{description}"
)
# stage 2: segment the object given a <loc> click
SEG_TEMPLATE = (
    "<image>\n Please output the segmentation mask of this object <loc> "
    "according to the following description. \n{description}"
)


def ponder_collate_fn(batch, max_point=-1):
    """Collate for point cloud; 'coord' is necessary to determine 'offset'."""
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")
    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for data in batch:
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


def build_pc_inputs(source, scan_folder, device):
    """Load + transform a scan into the tensors model.generate expects."""
    scan_file = source['scene_id']
    scan_data_path = pathlib.Path(scan_folder) / f'{scan_file}.pth'
    superpoint_path = pathlib.Path(scan_folder) / '../super_points' / f'{scan_file}.bin'

    raw_data = torch.load(scan_data_path)
    coord = raw_data['coord']
    color = raw_data['color']
    superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)
    instance = raw_data['instance_gt']

    transform = Compose(referseg_transform_eval)
    pc_data_dict = dict(coord=coord, color=color, superpoint_mask=superpoint_mask)
    pc_data_dict = transform(pc_data_dict)

    grid_coord = pc_data_dict['grid_coord']
    grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
    pc_data_dict['grid_coord'] = grid_coord

    grid_coords = pc_data_dict["grid_coord"]
    spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).numpy(), 128, None)
    voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, 1, 4)

    for key in pc_data_dict:
        if key in ["coord", "grid_coord", "feat", "offset"]:
            pc_data_dict[key] = ponder_collate_fn([pc_data_dict[key]])

    inputs = dict(
        coord=pc_data_dict["coord"].to(device, dtype=torch.bfloat16),
        grid_coord=voxel_coords.to(device),
        offset=pc_data_dict["offset"].to(device),
        feat=pc_data_dict["feat"].to(device, dtype=torch.bfloat16),
        p2v_map=p2v_map.to(device),
        v2p_map=v2p_map.to(device),
        spatial_shape=spatial_shape,
        superpoint_mask=[torch.tensor(superpoint_mask).to(device)],
    )
    return inputs, instance, superpoint_mask


def center_to_click_mask(pred_center, coord, superpoint_mask_tensor, num_click_sp, device):
    """Convert a predicted 3D center into a boolean superpoint click_mask by
    taking the nearest superpoint(s). Both pred_center and the per-superpoint
    centroids are in the same post-transform coord frame.

    coord is offset-packed (N, 3) with no batch dimension; superpoint_mask_tensor
    is the list-wrapped (N,) per-point superpoint ids.
    """
    pts = coord.float()                                      # (N, 3)
    sp_ids = superpoint_mask_tensor[0].long()                # (N,)
    sp_xyz = scatter(pts, sp_ids, reduce="mean", dim=0)      # (num_sp, 3)
    dists = torch.norm(sp_xyz.float() - pred_center.float().to(sp_xyz), dim=1)
    k = min(num_click_sp, sp_xyz.shape[0])
    topk = torch.topk(dists, k, largest=False).indices
    click_mask = torch.zeros(sp_xyz.shape[0], dtype=torch.bool, device=device)
    click_mask[topk] = True
    return click_mask


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, args.model_base, model_name,
        pointcloud_tower_name=args.pointcloud_tower_name)

    with open(args.question_file, 'r') as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    device = model.device
    for idx, source in enumerate(tqdm(questions)):
        inputs, instance, superpoint_mask = build_pc_inputs(source, args.scan_folder, device)
        object_id = int(source["object_id"])
        gt_mask = (instance == object_id).astype(bool)

        description = source['description']

        # ---- Stage 1: grounding -> predicted object center ----
        g_prompt = GROUNDING_TEMPLATE.format(description=description)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], g_prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_special_token(conv.get_prompt(), tokenizer, return_tensors='pt').unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_center = model.generate(
                input_ids,
                coord=inputs["coord"], grid_coord=inputs["grid_coord"], offset=inputs["offset"],
                feat=inputs["feat"], p2v_map=inputs["p2v_map"], v2p_map=inputs["v2p_map"],
                spatial_shape=inputs["spatial_shape"], superpoint_mask=inputs["superpoint_mask"],
                conditions=["grounding"],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
                max_new_tokens=64, tokenizer=tokenizer,
                click_mask=[[]], use_cache=True)
        # pred_center: (3,) in the post-transform coord frame

        # ---- convert predicted center -> superpoint click ----
        click_mask = center_to_click_mask(
            pred_center, inputs["coord"], inputs["superpoint_mask"], args.num_click_sp, device)

        # ---- Stage 2: segmentation with the predicted click ----
        s_prompt = SEG_TEMPLATE.format(description=description)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], s_prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_special_token(conv.get_prompt(), tokenizer, return_tensors='pt').unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_mask = model.generate(
                input_ids,
                coord=inputs["coord"], grid_coord=inputs["grid_coord"], offset=inputs["offset"],
                feat=inputs["feat"], p2v_map=inputs["p2v_map"], v2p_map=inputs["v2p_map"],
                spatial_shape=inputs["spatial_shape"], superpoint_mask=inputs["superpoint_mask"],
                conditions=["refer_seg"],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
                max_new_tokens=64, tokenizer=tokenizer,
                click_mask=[click_mask], use_cache=True)

        pred_mask = pred_mask.cpu().numpy().astype(bool)[0]
        I = np.sum(np.logical_and(pred_mask, gt_mask))
        U = np.sum(np.logical_or(pred_mask, gt_mask))
        iou = float(0) if U == 0 else float(I) / float(U)

        ans_file.write(json.dumps({
            "scene_id": source['scene_id'], "question_id": idx,
            "description": description, "model_id": model_name,
            "pred_center": pred_center.float().cpu().tolist(),
            "iou": iou, "tp50": int(iou >= 0.5), "tp25": int(iou >= 0.25),
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--pointcloud-tower-name", type=str, default=None)
    parser.add_argument("--scan-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,
                        default="playground/data/train_info/scanrefer_train_3d_llava_loc_predict.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num_click_sp", type=int, default=1,
                        help="number of nearest superpoints to select as the click")
    args = parser.parse_args()
    eval_model(args)
