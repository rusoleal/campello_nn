# Model fixture provenance

- `yunet_n_320_320.onnx` — from
  [ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train),
  `onnx/yunet_n_320_320.onnx`. BSD 3-Clause License (Copyright (c) 2022-2026, Shiqi Yu).
  Used here to test `campello_nn`'s ONNX importer against a real, standard face-detection
  model — see `tests/universal/test_yunet_face_detection.cpp`.

See `images/NOTICE.md` for the test image fixtures' provenance.
