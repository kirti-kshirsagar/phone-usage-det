# Phone Usage Detector

A computer vision system that detects when phones are actively held in hand from video footage. The system uses YOLO object detection and MediaPipe hand/pose tracking to identify phones that are being held by people.

## Features

- **Accurate Detection**: Detects phones only when held in hand, near face, or in lap
- **No False Positives**: Ignores static phones on tables or idle phones
- **Clean Output**: Simple "phone" labels with bounding boxes
- **Video Processing**: Maintains original resolution, FPS, and audio
- **Summary Report**: Generates total phone usage time statistics

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- MediaPipe
- PyTorch
- FFmpeg

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd phone-usage-detector
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python phone_usage_detector.py --input video.mp4 --output annotated_video.mp4 --summary summary.txt
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **Required** | Input video file (MP4, AVI, MOV) |
| `--output` | str | **Required** | Output annotated video file |
| `--summary` | str | None | Path for summary text file |
| `--model` | str | yolov8n.pt | YOLO model (n/s/m/l/x) |
| `--conf` | float | 0.15 | YOLO confidence threshold |
| `--device` | str | cpu | Inference device (cpu/cuda/mps) |
| `--proximity` | int | 60 | Pixels to consider "near hand" |
| `--max_age` | int | 30 | Track stale age (frames) |
| `--max_dist` | int | 80 | Track match max center distance |
| `--min_speed` | float | 1.5 | Min px/frame to be "moving" |
| `--skip` | int | 0 | Process every (skip+1)th frame for speed |

### Examples

**Process a video with default settings:**
```bash
python phone_usage_detector.py \
  --input sample_data/demo.mp4 \
  --output out/annotated.mp4 \
  --summary out/summary.txt
```

**Use GPU acceleration (if available):**
```bash
python phone_usage_detector.py \
  --input video.mp4 \
  --output output.mp4 \
  --device cuda \
  --summary summary.txt
```

**Adjust detection sensitivity:**
```bash
python phone_usage_detector.py \
  --input video.mp4 \
  --output output.mp4 \
  --conf 0.5 \
  --proximity 40 \
  --summary summary.txt
```

## How It Works

### Detection Pipeline

1. **Object Detection**: YOLO detects people and phones in each frame
2. **Duplicate Removal**: Non-Maximum Suppression (NMS) removes overlapping phone detections
3. **Hand Tracking**: MediaPipe tracks hand landmarks and pose
4. **Proximity Analysis**: Calculates distance between phones and hands/face
5. **Activity Detection**: Determines if phone is actively held based on:
   - Near hands (held in hand)
   - Near face (held up to face)
   - In lap AND near hands (held in lap)
6. **Track Deduplication**: Removes duplicate tracks that are too close together
7. **Temporal Smoothing**: Reduces flickering with frame-based smoothing
8. **Output Generation**: Creates annotated video and summary

### Detection Criteria

The system detects phones as "active" when they are:
- ✅ **Near hands** (within proximity threshold)
- ✅ **Near face** (intersecting head bounding box)
- ✅ **In lap AND near hands** (held in lap area)

The system ignores:
- ❌ Static phones on tables
- ❌ Phones away from person
- ❌ Idle phones in pockets

### Duplicate Detection Prevention

The system includes advanced duplicate removal:
- **Detection-level NMS**: Removes overlapping phone detections (40% IoU threshold)
- **Track-level deduplication**: Removes duplicate tracks within 60 pixels
- **Confidence-based selection**: Keeps the highest confidence detection when duplicates occur

## Output

### Annotated Video
- Green bounding boxes around detected phones
- "phone" labels with confidence scores
- Maintains original video quality and audio

### Summary File
```
Video: demo.mp4
FPS: 24.000 | Resolution: 3840x2160
Total active phone time: 9.46 seconds
```

## Performance

- **Processing Speed**: ~22-24 fps on CPU (varies by video resolution)
- **Accuracy**: High precision with minimal false positives
- **Memory Usage**: Efficient processing with frame skipping option

## Troubleshooting

### Common Issues

**"Failed to open input video"**
- Ensure video file exists and is readable
- Check video format (MP4, AVI, MOV supported)

**"Model must include 'person' and 'cell phone' classes"**
- Use YOLOv8 models (yolov8n.pt, yolov8s.pt, etc.)
- These include COCO dataset classes

**Slow processing**
- Use `--skip 1` to process every other frame
- Use `--device cuda` if GPU available
- Use smaller YOLO model (yolov8n.pt)


## Technical Details

### Models Used
- **YOLOv8**: Object detection (people, phones)
- **MediaPipe Hands**: Hand landmark tracking
- **MediaPipe Pose**: Body pose estimation

### Key Functions
- `is_active_phone_use()`: Core detection logic
- `track_speed()`: Motion analysis
- `SimpleTracker`: Object tracking across frames
- `extract_audio_to_temp()`: Audio preservation

### File Structure
```
phone-usage-detector/
├── phone_usage_detector.py    # Main script
├── requirements.txt           # Dependencies
├── README.md                 # This file
├── sample_data/              # Example videos
│   ├── demo.mp4
│   └── demo1.mp4
└── out/                      # Output directory
    ├── annotated.mp4
    └── summary.txt
```


## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
