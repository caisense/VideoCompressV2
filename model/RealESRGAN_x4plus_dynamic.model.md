# RealESRGAN_x4plus_dynamic.onnx

- Model: Real-ESRGAN x4plus, RRDBNet with 23 blocks, for general images.
- Runtime contract: float32 `input[1,3,H,W]` to float32 `output[1,3,4H,4W]`, opset 17.
- Source: [SkillSafe dynamic ONNX export](https://huggingface.co/skillsafe-ai/realesrgan-x4plus), converted from the upstream [Real-ESRGAN x4plus weights](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth).
- SHA-256: `4851ec156207d271f5328605d0582eeb851e656227da8aca093ced9e60789291`.
- Upstream weights and architecture license: BSD-3-Clause; copyright Xintao Wang, 2021.

The dynamic ONNX export was parity-checked against the pinned PyTorch weights by
the exporter. The runtime-specific TensorRT FP16 result still depends on the
GPU, TensorRT version, and selected execution providers.
