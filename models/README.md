# Models

Durable model checkpoints belong in `models/checkpoints/`.

PhoneWatch loads models in this order:

1. `models/checkpoints/phonewatch_best.pt`
2. `models/checkpoints/yolov8n.pt`
3. Ultralytics' default `yolov8n.pt` resolution path

Do not commit ad hoc exports or benchmark models unless they are intentionally part of the project release.
