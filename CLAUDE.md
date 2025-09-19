# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SenseVoice is a speech foundation model with multiple capabilities:
- **ASR (Automatic Speech Recognition)** - Multilingual speech recognition for 50+ languages
- **LID (Language Identification)** - Detects language automatically
- **SER (Speech Emotion Recognition)** - Classifies emotions: HAPPY, SAD, ANGRY, NEUTRAL, FEARFUL, DISGUSTED, SURPRISED
- **AED (Audio Event Detection)** - Detects events: BGM, Speech, Applause, Laughter, Cry, Sneeze, Breath, Cough

The model uses a non-autoregressive end-to-end architecture for extremely low inference latency (15x faster than Whisper-Large).

## Common Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
- **WebUI**: `python webui.py` - Gradio-based web interface with audio/video upload
- **API Server**: `fastapi run --port 50000` or `python api.py` - REST API service
- **Basic Testing**: `python demo1.py` or `python demo2.py`

### Model Export and Optimization
- **ONNX Export**: `python demo_onnx.py` - Export to ONNX format
- **LibTorch Export**: `python demo_libtorch.py` - Export to LibTorch format
- **General Export**: `python export.py` and `python export_meta.py`

### Fine-tuning
```bash
# First install FunASR for training
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./

# Run fine-tuning
bash finetune.sh
```

### Environment Configuration
- **GPU Device**: `export SENSEVOICE_DEVICE=cuda:0`
- **Multi-GPU Training**: Set `CUDA_VISIBLE_DEVICES="0,1"` in finetune.sh

## Code Architecture

### Core Components

**Main Model** (`model.py`):
- `SenseVoiceSmall` class - Primary model implementation
- Non-autoregressive architecture with SANM (Self-Attention with Normalization and Memory)
- Supports direct inference and FunASR integration

**Interface Layers**:
- `webui.py` - Gradio web interface supporting audio/video files and microphone input
- `api.py` - FastAPI REST service for production deployment
- Audio extraction functions are separated from transcription for better modularity

**Utilities** (`utils/`):
- `frontend.py` - Audio preprocessing and feature extraction
- `infer_utils.py` - Inference utilities and post-processing
- `ctc_alignment.py` - CTC forced alignment for timestamp generation
- `export_utils.py` - Model export functionality

### Data Processing

**Training Data Format** (JSONL):
- `key` - Unique audio file ID
- `source` - Audio file path
- `target` - Transcription text
- `text_language` - Language tags: `<|zh|>`, `<|en|>`, `<|yue|>`, `<|ja|>`, `<|ko|>`
- `emo_target` - Emotion labels with special tokens
- `event_target` - Event labels with special tokens
- `with_or_wo_itn` - Whether to include punctuation/ITN

**Audio Processing**:
- Supports any audio format via librosa
- Video processing via FFmpeg (webui.py extracts audio from video)
- Automatic resampling to 16kHz mono
- VAD (Voice Activity Detection) for long audio segmentation

### Model Configuration

**Key Parameters**:
- `trust_remote_code=True` - Uses local model.py instead of FunASR internal version
- `vad_model="fsmn-vad"` - VAD for long audio processing
- `batch_size_s=60` - Dynamic batching by audio duration
- `merge_vad=True` - Merge short VAD segments
- `use_itn=True` - Include punctuation and inverse text normalization

**Performance Tuning**:
- Remove VAD for short audio batches (<30s) to improve speed
- Use `batch_size` instead of `batch_size_s` for fixed batching
- DeepSpeed configuration available in `deepspeed_conf/ds_stage1.json`

## Development Guidelines

### Adding New Features
- Follow the separation of concerns: audio processing functions are separate from transcription
- Use the existing FunASR integration patterns for model loading
- Maintain compatibility with both direct model inference and FunASR AutoModel

### Testing
- Use example audio files in `example/` directory for testing different languages and scenarios
- Test both short audio (<30s) and long audio (>30s) scenarios
- Verify video processing works with supported formats: MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V

### Model Integration
- Always use `trust_remote_code=True` when loading models to use local modifications
- For production deployment, consider ONNX or LibTorch export for optimization
- Use appropriate VAD settings based on audio length and use case

### Audio Processing Notes
- The webui.py has been refactored to separate audio extraction from transcription
- Video files automatically trigger audio extraction via FFmpeg
- Proper error handling and cleanup for temporary files is implemented
- Support for both file uploads and microphone input

### Fine-tuning Data Preparation
Use `sensevoice2jsonl` command to convert SCP/text files to JSONL format:
```bash
sensevoice2jsonl \
++scp_file_list='["train_wav.scp", "train_text.txt", "train_text_language.txt", "train_emo.txt", "train_event.txt"]' \
++data_type_list='["source", "target", "text_language", "emo_target", "event_target"]' \
++jsonl_file_out="train.jsonl"
```