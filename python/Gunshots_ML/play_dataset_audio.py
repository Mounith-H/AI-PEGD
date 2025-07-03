#!/usr/bin/env python3
"""
Audio file player for testing gunshot detection
Plays audio files from your dataset through speakers to test the real-time detection
"""

import os
import librosa
import sounddevice as sd
import time
import random

# Dataset path
DATASET_PATH = "drive-download-20250702T171625Z-1-001"

def play_audio_file(file_path, sample_rate=22050):
    """Load and play an audio file"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)
        
        print(f"🔊 Playing: {os.path.basename(file_path)}")
        print(f"   Duration: {len(audio)/sr:.2f}s")
        
        # Play the audio
        sd.play(audio, sr)
        sd.wait()  # Wait until the file is done playing
        
        return True
    except Exception as e:
        print(f"❌ Error playing {file_path}: {e}")
        return False

def test_with_dataset_files():
    """Test the real-time detection with actual dataset files"""
    print("🎵 Dataset Audio File Player")
    print("=" * 40)
    print("This will play audio files from your dataset to test detection.\n")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
    
    # Get environment and gunshot files
    env_dir = os.path.join(DATASET_PATH, "environment_audio")
    gun_dir = os.path.join(DATASET_PATH, "gunshots_audio")
    
    env_files = []
    gun_files = []
    
    if os.path.exists(env_dir):
        env_files = [os.path.join(env_dir, f) for f in os.listdir(env_dir) if f.endswith(".wav")]
    
    if os.path.exists(gun_dir):
        gun_files = [os.path.join(gun_dir, f) for f in os.listdir(gun_dir) if f.endswith(".wav")]
    
    print(f"Found {len(env_files)} environment files and {len(gun_files)} gunshot files\n")
    
    if len(env_files) == 0 and len(gun_files) == 0:
        print("❌ No audio files found in dataset")
        return
    
    # Test with environment sounds
    if env_files:
        print("🌿 Testing with Environment Sounds:")
        print("-" * 35)
        for i, file_path in enumerate(random.sample(env_files, min(3, len(env_files)))):
            print(f"Test {i+1}/3:")
            if play_audio_file(file_path):
                time.sleep(2)  # Wait for detection processing
            print()
    
    # Test with gunshot sounds
    if gun_files:
        print("🔫 Testing with Gunshot Sounds:")
        print("-" * 30)
        for i, file_path in enumerate(random.sample(gun_files, min(3, len(gun_files)))):
            print(f"Test {i+1}/3:")
            if play_audio_file(file_path):
                time.sleep(2)  # Wait for detection processing
            print()
    
    print("✅ Dataset testing completed!")
    print("Check the realtime_detection.py output for detection results.")

def interactive_player():
    """Interactive audio file player"""
    print("🎮 Interactive Audio Player")
    print("=" * 25)
    
    env_dir = os.path.join(DATASET_PATH, "environment_audio")
    gun_dir = os.path.join(DATASET_PATH, "gunshots_audio")
    
    while True:
        print("\nChoose an option:")
        print("1. Play random environment sound")
        print("2. Play random gunshot sound")
        print("3. Run automated test")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            if os.path.exists(env_dir):
                env_files = [f for f in os.listdir(env_dir) if f.endswith(".wav")]
                if env_files:
                    file_path = os.path.join(env_dir, random.choice(env_files))
                    play_audio_file(file_path)
                    time.sleep(2)
                else:
                    print("❌ No environment files found")
            else:
                print("❌ Environment directory not found")
        
        elif choice == "2":
            if os.path.exists(gun_dir):
                gun_files = [f for f in os.listdir(gun_dir) if f.endswith(".wav")]
                if gun_files:
                    file_path = os.path.join(gun_dir, random.choice(gun_files))
                    play_audio_file(file_path)
                    time.sleep(2)
                else:
                    print("❌ No gunshot files found")
            else:
                print("❌ Gunshot directory not found")
        
        elif choice == "3":
            test_with_dataset_files()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    try:
        print("🎧 Make sure realtime_detection.py is running in another terminal!")
        print("This will play audio files for testing.\n")
        
        # Run interactive player
        interactive_player()
        
    except KeyboardInterrupt:
        print("\n🛑 Player stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
