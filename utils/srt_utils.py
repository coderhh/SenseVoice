# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
# SRT utilities for SenseVoice
# Enhanced with subtitle generation capabilities

import re
import os
from typing import List, Tuple, Optional
from datetime import timedelta


def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    total_seconds = seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    millisecs = int(round((total_seconds % 1) * 1000))

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def clean_text_for_srt(text: str) -> str:
    """Clean text for SRT format by removing special tokens and formatting"""
    # Remove emotion tokens like <|HAPPY|>, <|SAD|>, etc.
    text = re.sub(r'<\|[A-Z_]+\|>', '', text)

    # Remove language tokens like <|zh|>, <|en|>, etc.
    text = re.sub(r'<\|[a-z]+\|>', '', text)

    # Remove other special tokens
    text = re.sub(r'<\|[^>]+\|>', '', text)

    # Remove emojis for cleaner subtitle text (optional)
    # Uncomment the line below if you want to remove emojis
    # text = re.sub(r'[😊😔😡😰🤢😮🎼👏😀😭🤧😷❓]', '', text)

    # Clean up extra spaces
    text = ' '.join(text.split())

    return text.strip()


def split_long_subtitle(text: str, max_chars: int = 80, max_lines: int = 2) -> List[str]:
    """Split long subtitle text into multiple lines for better readability"""
    if len(text) <= max_chars:
        return [text]

    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= max_chars:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Word is too long, force break
                lines.append(word)
                current_line = ""

    if current_line:
        lines.append(current_line)

    # Limit to max_lines
    if len(lines) > max_lines:
        # Join excess lines to the last allowed line
        lines = lines[:max_lines-1] + [" ".join(lines[max_lines-1:])]

    return lines


def generate_srt_content(timestamps: List[Tuple[float, float]], texts: List[str],
                        max_chars: int = 80, max_lines: int = 2) -> str:
    """Generate SRT subtitle content from timestamps and texts"""
    if len(timestamps) != len(texts):
        raise ValueError(f"Timestamps ({len(timestamps)}) and texts ({len(texts)}) must have the same length")

    srt_content = []
    subtitle_index = 1

    for i, ((start_time, end_time), text) in enumerate(zip(timestamps, texts)):
        # Clean the text
        clean_text = clean_text_for_srt(text)

        if not clean_text:  # Skip empty subtitles
            continue

        # Split long subtitles
        subtitle_lines = split_long_subtitle(clean_text, max_chars, max_lines)

        # Format timestamps
        start_srt = format_timestamp_srt(start_time)
        end_srt = format_timestamp_srt(end_time)

        # Create subtitle entry
        subtitle_entry = [
            str(subtitle_index),
            f"{start_srt} --> {end_srt}",
            "\n".join(subtitle_lines),
            ""  # Empty line separator
        ]

        srt_content.extend(subtitle_entry)
        subtitle_index += 1

    return "\n".join(srt_content)


def parse_sensevoice_timestamps(timestamp_data) -> List[Tuple[float, float]]:
    """Parse SenseVoice timestamp format to start/end time pairs"""
    timestamps = []

    if isinstance(timestamp_data, list):
        for i in range(0, len(timestamp_data), 2):
            if i + 1 < len(timestamp_data):
                start_time = timestamp_data[i] / 1000.0  # Convert ms to seconds
                end_time = timestamp_data[i + 1] / 1000.0
                timestamps.append((start_time, end_time))
            else:
                # Handle odd number of timestamps
                start_time = timestamp_data[i] / 1000.0
                end_time = start_time + 1.0  # Default 1 second duration
                timestamps.append((start_time, end_time))

    return timestamps


def save_srt_file(srt_content: str, output_path: str) -> bool:
    """Save SRT content to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        return True
    except Exception as e:
        print(f"Error saving SRT file: {e}")
        return False


def create_srt_from_sensevoice_result(result, output_path: str,
                                    max_chars: int = 80, max_lines: int = 2) -> bool:
    """Create SRT file from SenseVoice inference result"""
    try:
        # Extract text and timestamps from result
        if isinstance(result, list) and len(result) > 0:
            res_data = result[0]
            if isinstance(res_data, list) and len(res_data) > 0:
                res_item = res_data[0]

                text = res_item.get("text", "")
                timestamp_data = res_item.get("timestamp", [])

                if not text or not timestamp_data:
                    print("No text or timestamp data found in result")
                    return False

                # For now, create a single subtitle for the entire text
                # This can be enhanced to split by sentences or segments
                clean_text = clean_text_for_srt(text)
                timestamps = parse_sensevoice_timestamps(timestamp_data)

                if not timestamps:
                    # Create default timestamp if none available
                    timestamps = [(0.0, 5.0)]  # Default 5-second subtitle

                # Use the first and last timestamp for the full text
                start_time = timestamps[0][0] if timestamps else 0.0
                end_time = timestamps[-1][1] if timestamps else 5.0

                srt_content = generate_srt_content(
                    [(start_time, end_time)],
                    [clean_text],
                    max_chars,
                    max_lines
                )

                return save_srt_file(srt_content, output_path)
            else:
                print("Invalid result format: expected list with data")
                return False
        else:
            print("Invalid result format: expected list")
            return False

    except Exception as e:
        print(f"Error creating SRT file: {e}")
        return False


def enhance_srt_with_speaker_labels(srt_content: str, speaker_labels: Optional[List[str]] = None) -> str:
    """Enhance SRT content with speaker labels (future enhancement)"""
    # This is a placeholder for future speaker diarization integration
    if not speaker_labels:
        return srt_content

    # TODO: Implement speaker label integration
    return srt_content


# Example usage and testing functions
def test_srt_generation():
    """Test SRT generation functionality"""
    # Test data
    test_timestamps = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.5)]
    test_texts = [
        "Hello, this is the first subtitle.",
        "This is the second subtitle with some <|HAPPY|> emotion tokens.",
        "And this is a very long third subtitle that should be split into multiple lines for better readability on screen."
    ]

    srt_content = generate_srt_content(test_timestamps, test_texts)
    print("Generated SRT content:")
    print(srt_content)

    return srt_content


if __name__ == "__main__":
    test_srt_generation()