#!/usr/bin/env python3
"""
Test script for real-time gunshot detection
This script helps you test the detection system with various sounds
"""

import time
import os
import numpy as np
import sounddevice as sd

def play_test_sound(frequency=440, duration=1.0, sample_rate=22050):
    """Generate and play a test sound"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate different types of test sounds
    sound = np.sin(2 * np.pi * frequency * t) * 0.3  # Basic sine wave
    
    print(f"🔊 Playing test sound: {frequency}Hz for {duration}s")
    sd.play(sound, sample_rate)
    sd.wait()  # Wait until sound is finished

def test_detection_system():
    """Run various test sounds to test the detection system"""
    print("🧪 Gunshot Detection Test Suite")
    print("=" * 40)
    print("Make sure realtime_detection.py is running in another terminal!")
    print("This script will play various test sounds...\n")
    
    # Test 1: Low frequency sound (environment-like)
    print("Test 1: Low frequency environment sound")
    play_test_sound(frequency=100, duration=2.0)
    time.sleep(3)
    
    # Test 2: Sharp high-frequency burst (gunshot-like)
    print("Test 2: Sharp high-frequency burst (gunshot-like)")
    play_test_sound(frequency=2000, duration=0.1)
    time.sleep(1)
    play_test_sound(frequency=1500, duration=0.05)
    time.sleep(3)
    
    # Test 3: Gunshot-like sound pattern
    print("Test 3: Simulated gunshot pattern")
    # Create a sharp attack followed by decay
    sample_rate = 22050
    duration = 0.2
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Sharp attack with multiple frequencies (like a gunshot)
    sound = (np.sin(2 * np.pi * 800 * t) + 
             np.sin(2 * np.pi * 1200 * t) + 
             np.sin(2 * np.pi * 1600 * t)) * np.exp(-t * 10) * 0.5
    
    print("🔊 Playing simulated gunshot sound")
    sd.play(sound, sample_rate)
    sd.wait()
    time.sleep(3)
    
    # Test 4: Clapping sound
    print("Test 4: Clapping sound")
    print("👏 Clap your hands loudly now!")
    time.sleep(5)
    
    # Test 5: Speaking
    print("Test 5: Speaking test")
    print("🗣️  Say something loudly!")
    time.sleep(5)
    
    print("\n✅ Test suite completed!")
    print("Check the realtime_detection.py output for results.")

if __name__ == "__main__":
    try:
        test_detection_system()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure sounddevice is installed: pip install sounddevice")
