# HEVC HM evaluation for machine-oriented BD-rate

This evaluation path is independent from `train_vcm_final.py`. It does not
change training weights, optimizer state, checkpoints, or the VCM loss.

## Required tools

- HEVC Test Model (HM) reference encoder, preferably HM-16.20 or HM-16.22.
- A matching HM configuration such as `encoder_lowdelay_P_main.cfg`.
- FFmpeg with PNG and raw YUV support.
- The same annotated full-resolution manifest used by `evaluate_vcm.py`.

Do not use NVENC as the reported HEVC reference anchor. Record the exact HM
version and configuration file with the results.

## Protocol choice

The project codec is P-frame-only and receives frame 0 as an external seed.
The matching HEVC mode is:

```text
--protocol conditional-pframes
```

HM still encodes frame 0 so that it can reconstruct the following pictures.
For the conditional comparison, `evaluate_hevc.py` excludes HM's reported POC
0 picture bits and excludes frame 0 from mAP. Sequence-level HEVC headers
remain counted. This result must be described as a conditional P-frame
comparison; it is not the rate of a standalone independently decodable HEVC
stream.

Use:

```text
--protocol all-frames
```

only when the candidate also transmits and counts its first frame.

## Smoke test

```bash
python evaluate_hevc.py \
  --data-dir /path/to/vcm_eval \
  --dataset-manifest manifest.json \
  --hm-encoder /path/to/HM/bin/TAppEncoderStatic \
  --hm-config /path/to/HM/cfg/encoder_lowdelay_P_main.cfg \
  --configuration-name "HM-16.22 Low-Delay P" \
  --protocol conditional-pframes \
  --qps 22 27 32 37 \
  --method-name "HEVC HM-16.22 LDP" \
  --max-sequences 1 \
  --output-dir output/hevc_smoke
```

Inspect the saved HM log. The evaluator deliberately fails when it cannot
parse the POC 0 bit count required by the conditional protocol.

## Full HEVC evaluation

```bash
python evaluate_hevc.py \
  --data-dir /path/to/vcm_eval \
  --dataset-manifest manifest.json \
  --hm-encoder /path/to/HM/bin/TAppEncoderStatic \
  --hm-config /path/to/HM/cfg/encoder_lowdelay_P_main.cfg \
  --configuration-name "HM-16.22 Low-Delay P" \
  --protocol conditional-pframes \
  --qps 22 27 32 37 \
  --method-name "HEVC HM-16.22 LDP" \
  --detector-size 640 \
  --detector-batch-size 16 \
  --confidence-threshold 0.001 \
  --nms-iou-threshold 0.6 \
  --max-detections 300 \
  --output-dir output/hevc_evaluation
```

The evaluator saves an automatic progress checkpoint after every completely
evaluated sequence (encoding, decoding, and mAP). If a Colab session ends,
repeat the identical command with `--resume`; at most the sequence active at
the interruption is re-run. Keep `--output-dir`, `--bitstream-dir`, and
`--encoder-log-dir` on Google Drive if the runtime itself may be reset.

For offline YOLOv5, add:

```bash
--yolov5-repo /path/to/yolov5-v7 \
--yolov5-weights /path/to/yolov5s.pt
```

The result JSON contains:

- actual BPP and kbps;
- mAP@0.5 and mAP@[0.5:0.95];
- full stream bits and excluded seed bits per sequence;
- exact evaluated frame count;
- dataset fingerprint and detector configuration;
- HM configuration and rate provenance.

## Proposed-codec evaluation

Run `evaluate_vcm.py --mode codec` on the same data and with the same detector
arguments. Both result files must contain the same `evaluation_id`,
`detector_config`, `ground_truth`, `protocol`, and evaluated frame count.

## BD-rate comparison

`compare_codecs_bd_rate.py` now rejects missing or mismatched fairness
metadata. Video comparison defaults to kbps:

```bash
python compare_codecs_bd_rate.py \
  --hevc-results output/hevc_evaluation/HEVC_HM-16.22_LDP_results.json \
  --learned-scalable-results output/learned_scalable_results.json \
  --proposed-results output/evaluation/dcvc_rt_vcm_results.json \
  --rate kbps \
  --metrics map50 map5095 \
  --output-dir output/codec_comparison
```

All four points must be Pareto-optimal and the curves must overlap in mAP.
A negative proposed-versus-HEVC BD-rate means that the proposed codec uses
less bitrate at equal machine-task accuracy.
