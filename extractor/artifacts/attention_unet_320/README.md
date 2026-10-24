# Frozen Attention U-Net candidate

This is an explicit, non-production inference artifact copied from
`extractor/experiments/final_pretrained320/attention_unet/best_checkpoint.pt`.
It uses an ImageNet-pretrained EfficientNet-B0 encoder, 320×320 input, ImageNet
mean/std normalization, and a segmentation threshold of 0.80. Load it by
passing `attention_unet_candidate.pt` explicitly to `ROIExtractor` or the
extractor CLI; the default `Models/UNet_extractor/unet.pt` remains unchanged.

`metadata.json` records the SHA-256, dependency versions, provenance, and
evaluation references. The test split contained 27 images, had no patient IDs,
and was used only once after the checkpoint and threshold were frozen; this is
not patient-independent validation.
