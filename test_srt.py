#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Test script for SRT generation functionality

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.srt_utils import (
    format_timestamp_srt,
    clean_text_for_srt,
    generate_srt_content,
    parse_sensevoice_timestamps,
    create_srt_from_sensevoice_result,
    test_srt_generation
)

def test_timestamp_formatting():
    """Test timestamp formatting function"""
    print("=== Testing Timestamp Formatting ===")
    test_times = [0.0, 1.5, 65.123, 3661.999]
    expected = ["00:00:00,000", "00:00:01,500", "00:01:05,123", "01:01:01,999"]

    for i, time_val in enumerate(test_times):
        result = format_timestamp_srt(time_val)
        print(f"Time {time_val}s -> {result} (expected: {expected[i]})")
        assert result == expected[i], f"Expected {expected[i]}, got {result}"

    print("Timestamp formatting tests passed!\n")

def test_text_cleaning():
    """Test text cleaning function"""
    print("=== Testing Text Cleaning ===")
    test_cases = [
        ("<|HAPPY|>Hello world<|zh|>", "Hello world"),
        ("This is <|SAD|> a test <|en|>", "This is a test"),
        ("<|BGM|>Music playing<|Event_UNK|>", "Music playing"),
        ("  Multiple   spaces  ", "Multiple spaces"),
        ("", "")
    ]

    for input_text, expected in test_cases:
        result = clean_text_for_srt(input_text)
        print(f"'{input_text}' -> '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print("Text cleaning tests passed!\n")

def test_full_srt_generation():
    """Test complete SRT generation"""
    print("=== Testing Full SRT Generation ===")

    # Test data
    timestamps = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.5)]
    texts = [
        "Hello, this is the first subtitle.",
        "This is the <|HAPPY|> second subtitle.",
        "And this is a very long third subtitle that should be split into multiple lines for better readability on screen when displayed."
    ]

    srt_content = generate_srt_content(timestamps, texts, max_chars=50, max_lines=2)

    print("Generated SRT Content:")
    print("-" * 50)
    print(srt_content)
    print("-" * 50)

    # Verify basic structure
    lines = srt_content.split('\n')
    assert '1' in lines, "Should contain subtitle index 1"
    assert '00:00:00,000 --> 00:00:02,500' in lines, "Should contain first timestamp"
    assert 'Hello, this is the first subtitle.' in lines, "Should contain first subtitle text"

    print("Full SRT generation test passed!\n")

def test_sensevoice_result_format():
    """Test SRT creation from mock SenseVoice result"""
    print("=== Testing SenseVoice Result Processing ===")

    # Mock SenseVoice result format
    mock_result = [
        [
            {
                "text": "<|en|>Hello world<|HAPPY|>",
                "timestamp": [0, 1000, 1000, 2500]  # Start, end, start, end in milliseconds
            }
        ]
    ]

    output_path = "test_output.srt"

    try:
        success = create_srt_from_sensevoice_result(mock_result, output_path)
        assert success, "SRT creation should succeed"

        # Verify file exists and has content
        assert os.path.exists(output_path), "SRT file should be created"

        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            print("Generated SRT from mock result:")
            print("-" * 40)
            print(content)
            print("-" * 40)

            assert '1' in content, "Should have subtitle index"
            assert '00:00:00,000 --> 00:00:02,500' in content, "Should have timestamp"
            assert 'Hello world' in content, "Should contain cleaned text"
            assert '<|en|>' not in content, "Should not contain language tokens"
            assert '<|HAPPY|>' not in content, "Should not contain emotion tokens"

        print("SenseVoice result processing test passed!\n")

    finally:
        # Clean up test file
        if os.path.exists(output_path):
            os.remove(output_path)

def main():
    """Run all tests"""
    print("Starting SRT Generation Tests\n")

    try:
        test_timestamp_formatting()
        test_text_cleaning()
        test_full_srt_generation()
        test_sensevoice_result_format()

        print("All tests passed successfully!")
        print("\nSRT generation functionality is ready to use!")

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()