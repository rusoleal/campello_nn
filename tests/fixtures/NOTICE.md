# Model fixture provenance

- `yunet_n_320_320.onnx` — from
  [ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train),
  `onnx/yunet_n_320_320.onnx`. BSD 3-Clause License (Copyright (c) 2022-2026, Shiqi Yu).
  Used here to test `campello_nn`'s ONNX importer against a real, standard face-detection
  model — see `tests/universal/test_yunet_face_detection.cpp`.

- `tflite_schema.fbs` — from
  [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT),
  `tflite/converter/schema/schema.fbs` (the canonical TFLite FlatBuffers schema; moved here
  during the TFLite->LiteRT migration, the old `tensorflow/tensorflow` path no longer has it).
  Apache License 2.0 (Copyright 2017 The TensorFlow Authors). Used only to regenerate
  `conv_add_relu.tflite` via `flatc --binary --schema tflite_schema.fbs --
  conv_add_relu_tflite.json` — not compiled into campello_nn itself (the real build fetches
  its own copy of the schema indirectly through whatever `flatc`/`flatbuffers` version is used
  to regenerate fixtures; `tflite_parser.cpp`'s hand-written field accessors are what the
  library actually ships with).

- `blaze_face_short_range.tflite` — from
  [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector),
  downloaded from
  `storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`
  (the official "float16" export — weights stored as FLOAT16, see
  `tests/universal/test_blazeface_face_detection.cpp` and `src/tflite/tflite_importer.cpp`'s
  `TFL_DEQUANTIZE` handling for why that matters). Apache License 2.0 (Copyright Google LLC),
  same as the rest of the MediaPipe project this model ships under — see the
  [MediaPipe BlazeFace Model Card (Short Range)](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf).
  Used to test `campello_nn`'s TFLite importer against a second, independent real model (the
  first being YuNet/ONNX above) — this one specifically exercises `DEPTHWISE_CONV_2D`, `PAD`
  (as `concat`-with-zeros — see that op's comment in `tflite_importer.cpp`), and float16-weight
  `DEQUANTIZE`, none of which YuNet's ONNX graph needed.

See `images/NOTICE.md` for the test image fixtures' provenance.
