# setting for transformation in data loader

scannet_axis_align_file = "playground/data/scannet/scannet_axis_align_matrix_trainval.pkl"

################################ Referring Segmentation ##########################
referseg_transform_train=[
    dict(type="CenterShift", apply_z=True),
    dict(
        type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
    ),
    dict(type="RandomFlip", p=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type="ElasticDistortionV2", 
            gran=[6, 20], 
            mag=[40, 160], 
            grid_size=0.02,
            p=0.5),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ShufflePoint"),
    dict(type="GetContinualSuperpointMask"),
    dict(type="Add", keys_dict={"condition": "refer_seg"}),
    dict(type="AddReferTarget"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "gt_masks_3d", "gt_labels_3d", "condition"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


#################### Referring Segmentation with Click (Ablation) ####################
referseg_transform_train_with_click=[
    dict(type="CenterShift", apply_z=True),
    dict(
        type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
    ),
    dict(type="RandomFlip", p=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type="ElasticDistortionV2",
            gran=[6, 20],
            mag=[40, 160],
            grid_size=0.02,
            p=0.5),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ShufflePoint"),
    dict(type="GetContinualSuperpointMask"),
    dict(type="Add", keys_dict={"condition": "refer_seg"}),
    dict(type="AddReferTarget"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "gt_masks_3d", "gt_labels_3d", "condition", "obj_sp_mask"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


############################## Grounding ([LOC] coordinate regression) ##############################
# Predicts the object 3D center (gt_loc) from a text description. gt_loc is
# computed in the post-augmentation coord frame so it matches the eval-time
# sp_xyz used for the nearest-superpoint click. AddGroundingTarget runs AFTER
# ToTensor (like Mask2Box) so coord is already a tensor.
grounding_transform_train=[
    dict(type="CenterShift", apply_z=True),
    dict(
        type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
    ),
    dict(type="RandomFlip", p=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type="ElasticDistortionV2",
            gran=[6, 20],
            mag=[40, 160],
            grid_size=0.02,
            p=0.5),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ShufflePoint"),
    dict(type="GetContinualSuperpointMask"),
    dict(type="Add", keys_dict={"condition": "grounding"}),
    dict(type="ToTensor"),
    dict(type="AddGroundingTarget"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "condition", "gt_loc"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


######################################## VQA ########################################
vqa_transform_train=[
    dict(type="SceneAlignment", file_path=scannet_axis_align_file),
    dict(type="CenterShift", apply_z=True),
    dict(
        type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.088, 0.088],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ShufflePoint"),
    dict(type="GetContinualSuperpointMask"),
    dict(type="Add", keys_dict={"condition": "vqa"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "condition"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


###################################### Dense Captioning ####################################
densecap_transform_train=[
    dict(type="SceneAlignment", file_path=scannet_axis_align_file),
    dict(type="Refer2InstanceMask"),
    dict(type="CenterShift", apply_z=True),
    dict(
        type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
    ),
    dict(type="RandomFlip", p=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.088, 0.088],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ShufflePoint"),
    dict(type="GetContinualSuperpointMask"),
    dict(type="ToTensor"),
    dict(type="Mask2Box", is_train=True),
    dict(type="Add", keys_dict={"condition": "dense_captioning"}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "condition", "obj_click", "obj_sp_mask"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


referseg_transform_eval=[
    dict(type="CenterShift", apply_z=True),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="Add", keys_dict={"condition": "refer_seg"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "condition"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


vqa_transform_eval=[
    dict(type="SceneAlignment", file_path=scannet_axis_align_file),
    dict(type="CenterShift", apply_z=True),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="Add", keys_dict={"condition": "textgen"}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "condition"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]


densecap_transform_eval=[
    dict(type="SceneAlignment", file_path=scannet_axis_align_file),
    dict(type="CenterShift", apply_z=True),
    dict(type="VoxelizationInfo", grid_size=0.02),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="Add", keys_dict={"condition": "textgen"}),
    dict(type="ToTensor"),
    dict(type="Mask2Box", is_train=True),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "superpoint_mask", "condition", "obj_click", "obj_sp_mask", "obj_box"),
        feat_keys=("coord", "color",),
        offset_keys_dict=dict(offset="coord")
    ),
]