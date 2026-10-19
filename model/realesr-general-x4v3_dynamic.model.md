# realesr-general-x4v3 dynamic ONNX

- Upstream: [Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0), BSD-3-Clause, copyright Xintao Wang.
- Architecture: SRVGGNetCompact, 64 features, 32 body convolutions, PReLU, PixelShuffle x4, nearest-neighbor residual. 1,213,296 parameters.
- Contract: FP32 RGB `input[1,3,H,W]` to FP32 RGB `output[1,3,4H,4W]`, ONNX opset 17. Model storage/input dtype is separate from requested TensorRT FP16 execution.
- Strong-denoise upstream checkpoint: `realesr-general-x4v3.pth`, SHA256 `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`.
- Weak-denoise upstream checkpoint: `realesr-general-wdn-x4v3.pth`, SHA256 `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca`.

Exported variants:

| File | Denoise strength | ONNX SHA256 |
|---|---:|---|
| `realesr-general-x4v3_dynamic.onnx` | 1.0, original strong weights | `192a94a79e5a47a85b51bd2b8756fcbb9f8e10813d823eef063db01c252e73bf` |
| `realesr-general-x4v3_dn05_dynamic.onnx` | 0.5, upstream CLI default: 0.5 strong + 0.5 weak weights | `1bbbb97b45b64d5f48750e7244521a6384be502d8a78b5efdef6d333d2328dee` |

Both pass ONNX checker and PyTorch-versus-ORT-CPU comparisons at two dynamic input shapes; maximum absolute FP32 discrepancy is below 6e-6. TensorRT FP16 differences are recorded separately in the test report; FP16 is not claimed to be numerically identical to FP32.

Export script, environment and validation records: `runs/gan/x4v3_test_20261016/`.
Select the experimental model with the existing `--gan-esrgan-x4-model` argument. The gan4 CAP120 profile outputs the full native 1920x1080 image, or 1080x1920 after ccw90 rotation. The default model path remains x4plus.
