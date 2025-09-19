# coding=utf-8

import os
import librosa
import base64
import io
import gradio as gr
import re
import tempfile
import subprocess
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
import torchaudio

from funasr import AutoModel
from utils.srt_utils import create_srt_from_sensevoice_result, format_timestamp_srt

model = "iic/SenseVoiceSmall"
model = AutoModel(model=model,
				  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
				  vad_kwargs={"max_single_segment_time": 30000},
				  trust_remote_code=True,
				  )

import re

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def browse_for_folder():
	"""Open a folder selection dialog and return the selected path"""
	try:
		# Create a root window and hide it
		root = tk.Tk()
		root.withdraw()
		root.attributes('-topmost', True)

		# Open directory selection dialog
		directory = filedialog.askdirectory(
			title="Select Output Directory for Extracted Audio",
			initialdir=os.path.expanduser("~")
		)

		# Clean up the root window
		root.destroy()

		return directory if directory else ""
	except Exception as e:
		print(f"[AUDIO EXTRACTION TAB] Error opening folder browser: {e}")
		return ""

def check_ffmpeg_available():
	"""Check if ffmpeg is available in the system"""
	try:
		result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
		return result.returncode == 0
	except FileNotFoundError:
		return False

def is_video_file(file_path):
	"""Check if the file is a video file based on extension"""
	if not isinstance(file_path, str):
		return False
	video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
	return any(file_path.lower().endswith(ext) for ext in video_extensions)

def extract_audio_from_video(video_path, output_path=None):
	"""Extract audio from video file using ffmpeg"""
	if not check_ffmpeg_available():
		raise Exception("FFmpeg is not available. Please install FFmpeg to extract audio from video files.")

	if not os.path.exists(video_path):
		raise Exception(f"Video file not found: {video_path}")

	# If no output path specified, create temporary file
	if output_path is None:
		temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
		temp_audio.close()
		audio_output_path = temp_audio.name
	else:
		audio_output_path = output_path

	try:
		# Use ffmpeg to extract audio
		cmd = [
			'ffmpeg', '-i', video_path,
			'-acodec', 'pcm_s16le',
			'-ac', '1',  # mono
			'-ar', '16000',  # 16kHz sample rate
			'-y',  # overwrite output file
			audio_output_path
		]

		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode != 0:
			raise Exception(f"FFmpeg error: {result.stderr}")

		return audio_output_path
	except Exception as e:
		# Clean up file if extraction failed and it was a temp file
		if output_path is None and os.path.exists(audio_output_path):
			try:
				os.unlink(audio_output_path)
			except:
				pass
		raise Exception(f"Error extracting audio from video: {e}")

def extract_audio_only(video_file, output_directory=""):
	"""Extract audio from video file and save to specified directory"""
	print(f"[AUDIO EXTRACTION TAB] Starting audio extraction...")

	if video_file is None:
		return None, "Please upload a video file"

	try:
		# Get the file path
		video_path = video_file.name if hasattr(video_file, 'name') else video_file
		print(f"[AUDIO EXTRACTION TAB] Processing video file: {video_path}")

		if not is_video_file(video_path):
			return None, "Please upload a valid video file (MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V)"

		# Get original filename from gradio file object or path
		if hasattr(video_file, 'orig_name') and video_file.orig_name:
			original_filename = video_file.orig_name
			print(f"[AUDIO EXTRACTION TAB] Original filename: {original_filename}")
		else:
			original_filename = os.path.basename(video_path)
			print(f"[AUDIO EXTRACTION TAB] Using temp filename: {original_filename}")

		# Determine output directory
		if output_directory and output_directory.strip():
			output_dir = output_directory.strip()
			print(f"[AUDIO EXTRACTION TAB] Using specified path: {output_dir}")
		else:
			# Default to Downloads folder if no directory specified
			output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
			print(f"[AUDIO EXTRACTION TAB] Using default Downloads folder: {output_dir}")

		# Create directory if it doesn't exist
		if not os.path.exists(output_dir):
			try:
				os.makedirs(output_dir)
				print(f"[AUDIO EXTRACTION TAB] Created directory: {output_dir}")
			except Exception as e:
				print(f"[AUDIO EXTRACTION TAB] Failed to create directory: {e}")
				return None, f"Failed to create directory: {output_dir}. Error: {str(e)}"

		print(f"[AUDIO EXTRACTION TAB] Using output directory: {output_dir}")

		video_name = os.path.splitext(original_filename)[0]
		audio_output_path = os.path.join(output_dir, f"{video_name}.wav")

		# Handle filename conflicts
		counter = 1
		base_audio_path = audio_output_path
		while os.path.exists(audio_output_path):
			audio_output_path = os.path.join(output_dir, f"{video_name}_{counter}.wav")
			counter += 1

		print(f"[AUDIO EXTRACTION TAB] Saving audio to: {audio_output_path}")

		# Extract audio to the specified location
		extract_audio_from_video(video_path, audio_output_path)

		print(f"[AUDIO EXTRACTION TAB] Audio extraction completed successfully")

		# Clean up temporary video file
		try:
			if os.path.exists(video_path):
				print(f"[AUDIO EXTRACTION TAB] Cleaning up temporary video file: {video_path}")
				os.unlink(video_path)
				print(f"[AUDIO EXTRACTION TAB] Temporary video file deleted successfully")
		except Exception as cleanup_error:
			print(f"[AUDIO EXTRACTION TAB] Failed to delete temporary video file: {cleanup_error}")

		return None, f"Audio successfully extracted and saved to: {audio_output_path}"

	except Exception as e:
		print(f"[AUDIO EXTRACTION TAB] Error: {str(e)}")
		# Clean up temporary video file even if extraction failed
		try:
			if 'video_path' in locals() and os.path.exists(video_path):
				print(f"[AUDIO EXTRACTION TAB] Cleaning up temporary video file after error: {video_path}")
				os.unlink(video_path)
		except Exception as cleanup_error:
			print(f"[AUDIO EXTRACTION TAB] Failed to delete temporary video file after error: {cleanup_error}")
		return None, f"Error extracting audio: {str(e)}"

def model_inference(input_wav, language, enable_timestamps=False, generate_srt=False, srt_output_dir="", fs=16000):
	# task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
	language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
					 "nospeech": "nospeech"}

	# task = "Speech Recognition" if task is None else task
	language = "auto" if len(language) < 1 else language
	selected_language = language_abbr[language]
	# selected_task = task_abbr.get(task)

	# print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")

	# Track extracted audio file for cleanup
	extracted_audio_path = None

	# Handle video files by extracting audio first
	if hasattr(input_wav, 'name') and is_video_file(input_wav.name):
		try:
			# Get the original video file path (Gradio temp path)
			video_path = input_wav.name
			print(f"[SPEECH RECOGNITION] Processing video file: {video_path}")

			# Get original video file info to determine where to save audio
			# For Gradio uploads, we'll use the temp directory but with better naming
			video_name = os.path.splitext(os.path.basename(video_path))[0]
			video_dir = os.path.dirname(video_path)

			# If the video_path looks like a Gradio temp file, try to get original name
			if hasattr(input_wav, 'orig_name') and input_wav.orig_name:
				original_name = os.path.splitext(input_wav.orig_name)[0]
				audio_output_path = os.path.join(video_dir, f"{original_name}.wav")
				print(f"[SPEECH RECOGNITION] Using original name: {input_wav.orig_name}")
			else:
				audio_output_path = os.path.join(video_dir, f"{video_name}.wav")
				print(f"[SPEECH RECOGNITION] Using temp name: {video_name}")

			extracted_audio_path = audio_output_path
			print(f"[SPEECH RECOGNITION] Extracting audio to: {audio_output_path}")

			# Extract audio to the specified location
			extract_audio_from_video(video_path, audio_output_path)
			print(f"[SPEECH RECOGNITION] Audio extraction completed")

			# Load the extracted audio file
			input_wav, fs = librosa.load(audio_output_path, sr=16000)
			print(f"[SPEECH RECOGNITION] Audio loaded for processing")

		except Exception as e:
			# Clean up extracted audio file if it was created
			if extracted_audio_path and os.path.exists(extracted_audio_path):
				try:
					os.unlink(extracted_audio_path)
				except:
					pass
			return f"Error processing video file: {str(e)}"

	if isinstance(input_wav, tuple):
		fs, input_wav = input_wav
		input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
		if len(input_wav.shape) > 1:
			input_wav = input_wav.mean(-1)
		if fs != 16000:
			print(f"audio_fs: {fs}")
			resampler = torchaudio.transforms.Resample(fs, 16000)
			input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
			input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
	
	
	merge_vad = True #False if selected_task == "ASR" else True
	print(f"language: {language}, merge_vad: {merge_vad}, timestamps: {enable_timestamps}")

	# Generate inference with optional timestamps
	inference_params = {
		"input": input_wav,
		"cache": {},
		"language": language,
		"use_itn": True,
		"batch_size_s": 60,
		"merge_vad": merge_vad
	}

	# Add timestamp support if requested
	if enable_timestamps:
		inference_params["output_timestamp"] = True
		print("[SPEECH RECOGNITION] Generating transcription with timestamps")

	result = model.generate(**inference_params)
	print(result)

	# Extract text
	text = result[0]["text"]
	text = format_str_v3(text)
	print(f"[SPEECH RECOGNITION] Transcription: {text}")

	# Generate SRT file if requested
	srt_status = ""
	if generate_srt and enable_timestamps:
		try:
			# Determine output directory for SRT
			if srt_output_dir and srt_output_dir.strip():
				output_dir = srt_output_dir.strip()
			else:
				output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

			# Ensure directory exists
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

			# Generate SRT filename
			import time
			timestamp_str = time.strftime("%Y%m%d_%H%M%S")
			srt_filename = f"transcription_{timestamp_str}.srt"
			srt_path = os.path.join(output_dir, srt_filename)

			# Create SRT file
			if create_srt_from_sensevoice_result(result, srt_path):
				srt_status = f"\n\nSRT file saved to: {srt_path}"
				print(f"[SPEECH RECOGNITION] SRT file created: {srt_path}")
			else:
				srt_status = "\n\nFailed to generate SRT file"
				print("[SPEECH RECOGNITION] Failed to create SRT file")

		except Exception as e:
			srt_status = f"\n\nError generating SRT: {str(e)}"
			print(f"[SPEECH RECOGNITION] SRT generation error: {e}")
	elif generate_srt and not enable_timestamps:
		srt_status = "\n\nNote: Enable timestamps to generate SRT files"

	# Format final output
	final_text = text + srt_status

	# Clean up extracted audio file after successful processing
	if extracted_audio_path and os.path.exists(extracted_audio_path):
		try:
			print(f"[SPEECH RECOGNITION] Cleaning up temporary audio file: {extracted_audio_path}")
			os.unlink(extracted_audio_path)
			print(f"[SPEECH RECOGNITION] Temporary audio file deleted successfully")
		except Exception as e:
			print(f"[SPEECH RECOGNITION] Failed to delete temporary audio file: {e}")

	return final_text


audio_examples = [
    ["example/zh.mp3", "zh"],
    ["example/yue.mp3", "yue"],
    ["example/en.mp3", "en"],
    ["example/ja.mp3", "ja"],
    ["example/ko.mp3", "ko"],
    ["example/emo_1.wav", "auto"],
    ["example/emo_2.wav", "auto"],
    ["example/emo_3.wav", "auto"],
    #["example/emo_4.wav", "auto"],
    #["example/event_1.wav", "auto"],
    #["example/event_2.wav", "auto"],
    #["example/event_3.wav", "auto"],
    ["example/rich_1.wav", "auto"],
    ["example/rich_2.wav", "auto"],
    #["example/rich_3.wav", "auto"],
    ["example/longwav_1.wav", "auto"],
    ["example/longwav_2.wav", "auto"],
    ["example/longwav_3.wav", "auto"],
    #["example/longwav_4.wav", "auto"],
]



html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Voice Understanding Model: SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small is an encoder-only speech foundation model designed for rapid voice understanding. It encompasses a variety of features including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and acoustic event detection (AED). SenseVoice-Small supports multilingual recognition for Chinese, English, Cantonese, Japanese, and Korean. Additionally, it offers exceptionally low inference latency, performing 7 times faster than Whisper-small and 17 times faster than Whisper-large.</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2> <p style="font-size: 18px;margin-left: 20px;">Upload an audio file or input through a microphone, then select the task and language. the audio is transcribed into corresponding text along with associated emotions (😊 happy, 😡 angry/exicting, 😔 sad) and types of sound events (😀 laughter, 🎼 music, 👏 applause, 🤧 cough&sneeze, 😭 cry). The event labels are placed in the front of the text and the emotion are in the back of the text.</p>
	<p style="font-size: 18px;margin-left: 20px;">Recommended audio input duration is below 30 seconds. For audio longer than 30 seconds, local deployment is recommended.</p>
	<h2 style="font-size: 22px;margin-left: 0px;">Repo</h2>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>: multilingual speech understanding model</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>: fundamental speech recognition toolkit</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice</a>: high-quality multilingual TTS model</p>
</div>
"""


def launch():
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		# gr.Markdown(description)
		gr.HTML(html_content)

		with gr.Tabs():
			with gr.Tab("Speech Recognition"):
				with gr.Row():
					with gr.Column():
						audio_inputs = gr.Audio(label="Upload audio or use the microphone")

						with gr.Accordion("Configuration"):
							language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
														  value="auto",
														  label="Language")

							with gr.Row():
								enable_timestamps = gr.Checkbox(
									label="Generate Timestamps",
									value=False,
									info="Enable timestamp generation for time-aligned transcription"
								)
								generate_srt = gr.Checkbox(
									label="Generate SRT Subtitles",
									value=False,
									info="Create SRT subtitle file (requires timestamps)"
								)

							srt_output_dir = gr.Textbox(
								label="SRT Output Directory (Optional)",
								placeholder="e.g., C:\\Subtitles (leave blank for Downloads folder)",
								info="Directory where SRT files will be saved. Defaults to Downloads if empty.",
								visible=False
							)

						fn_button = gr.Button("Start", variant="primary")
						text_outputs = gr.Textbox(label="Results", lines=5)
					gr.Examples(examples=audio_examples, inputs=[audio_inputs, language_inputs], examples_per_page=20)

				# Show/hide SRT directory input based on SRT checkbox
				def toggle_srt_directory(generate_srt_enabled):
					return gr.update(visible=generate_srt_enabled)

				generate_srt.change(toggle_srt_directory, inputs=[generate_srt], outputs=[srt_output_dir])

				fn_button.click(
					model_inference,
					inputs=[audio_inputs, language_inputs, enable_timestamps, generate_srt, srt_output_dir],
					outputs=text_outputs
				)

			with gr.Tab("Audio Extraction"):
				with gr.Row():
					with gr.Column():
						gr.Markdown("### Extract Audio from Video")
						gr.Markdown("Upload a video file to extract its audio track. You can specify a custom output directory or leave it blank to use the Downloads folder.")
						gr.Markdown("**Supported formats:** MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V")

						video_input = gr.File(
							label="Upload Video File",
							file_types=[".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"]
						)

						with gr.Row():
							output_directory = gr.Textbox(
								label="Output Directory (Optional)",
								placeholder="e.g., C:\\MyAudioFiles or D:\\Videos\\Audio (leave blank for Downloads folder)",
								info="Enter the full path or use the Browse button. If left blank, files will be saved to your Downloads folder.",
								scale=4
							)
							browse_folder_btn = gr.Button("Browse Folder", scale=1, variant="secondary")

						extract_button = gr.Button("Extract Audio", variant="primary")

						extraction_status = gr.Textbox(label="Status", interactive=False, lines=3)

				browse_folder_btn.click(
					browse_for_folder,
					outputs=[output_directory]
				)

				extract_button.click(
					extract_audio_only,
					inputs=[video_input, output_directory],
					outputs=[extraction_status]
				)

	demo.launch()


if __name__ == "__main__":
	# iface.launch()
	launch()


