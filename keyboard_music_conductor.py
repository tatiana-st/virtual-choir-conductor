#!/usr/bin/env python3
"""
Virtual Choir Conductor - Real-Time Music Player
Press arrow keys to play next note from sheet music.
Each beat triggers the next note/syllable.
"""

import cv2
import numpy as np
import time
import threading
import simpleaudio as sa
from collections import deque
from datetime import datetime
import os
import math

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    "beat_cooldown": 0.2,           # Min seconds between notes
    "time_signature": "4/4",
    "auto_advance": False,           # If True, auto-advance at current BPM
    "auto_bpm": 90,                  # BPM for auto-advance mode
}

# ============================================
# SIMPLE MUSIC NOTE GENERATOR
# ============================================
class NoteGenerator:
    """Generates sine wave tones for different notes"""
    
    # Frequency mapping (Hz)
    FREQUENCIES = {
        'C4': 261.63,
        'D4': 293.66,
        'E4': 329.63,
        'F4': 349.23,
        'G4': 392.00,
        'A4': 440.00,
        'B4': 493.88,
        'C5': 523.25,
        'D5': 587.33,
        'E5': 659.25,
        'F5': 698.46,
        'G5': 783.99,
        'A5': 880.00,
        'B5': 987.77,
    }
    
    # Choir syllables (for vocal simulation)
    SYLLABLES = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti']
    
    def __init__(self):
        self.sample_rate = 44100
        self.current_audio = None
        
    def generate_tone(self, frequency, duration=0.3, volume=0.3):
        """Generate a sine wave tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        wave = volume * np.sin(2 * np.pi * frequency * t)
        
        # Fade in/out to avoid clicks
        fade_samples = int(0.01 * self.sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
        
        # Convert to int16
        audio = (wave * 32767).astype(np.int16)
        return audio
    
    def play_note(self, note_name, duration=0.3):
        """Play a musical note"""
        if note_name in self.FREQUENCIES:
            freq = self.FREQUENCIES[note_name]
            audio = self.generate_tone(freq, duration)
            
            # Stop previous note if playing
            if self.current_audio:
                self.current_audio.stop()
            
            # Play new note
            self.current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)
            return True
        return False
    
    def play_chord(self, notes, duration=0.5):
        """Play multiple notes together (chord)"""
        if not notes:
            return
        
        # Generate combined wave
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        combined = np.zeros(len(t))
        
        for note in notes:
            if note in self.FREQUENCIES:
                freq = self.FREQUENCIES[note]
                combined += 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Normalize
        combined = combined / len(notes) if notes else combined
        combined = np.clip(combined, -0.5, 0.5)
        
        # Convert to int16
        audio = (combined * 32767).astype(np.int16)
        
        # Stop previous
        if self.current_audio:
            self.current_audio.stop()
        
        # Play
        self.current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)
    
    def play_beep(self):
        """Simple beep for beat feedback"""
        audio = self.generate_tone(880, 0.1, 0.2)
        self.current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)

# ============================================
# MUSIC SHEET (Score) - Simple Melodies
# ============================================
class MusicScore:
    """Manages the music sheet and current position"""
    
    # Simple choir melodies
    SCORES = {
        "major_scale": {
            "name": "Major Scale",
            "notes": ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
            "duration": 0.4,
            "type": "melody"
        },
        "ode_to_joy": {
            "name": "Ode to Joy",
            "notes": ['E4', 'E4', 'F4', 'G4', 'G4', 'F4', 'E4', 'D4', 
                     'C4', 'C4', 'D4', 'E4', 'E4', 'D4', 'D4'],
            "duration": 0.4,
            "type": "melody"
        },
        "choir_chord": {
            "name": "Choir Chord",
            "notes": [['C4', 'E4', 'G4'], ['D4', 'F4', 'A4'], ['E4', 'G4', 'B4'], ['F4', 'A4', 'C5']],
            "duration": 0.6,
            "type": "chord"
        },
        "simple_song": {
            "name": "Simple Song",
            "notes": ['C4', 'C4', 'G4', 'G4', 'A4', 'A4', 'G4', 
                     'F4', 'F4', 'E4', 'E4', 'D4', 'D4', 'C4'],
            "duration": 0.35,
            "type": "melody"
        },
        "silent_night": {
            "name": "Silent Night",
            "notes": ['G4', 'A4', 'G4', 'E4', 'G4', 'A4', 'G4', 'E4',
                     'D4', 'D4', 'B3', 'C4', 'D4', 'D4', 'B3', 'C4',
                     'G4', 'A4', 'G4', 'E4', 'G4', 'A4', 'G4', 'E4'],
            "duration": 0.5,
            "type": "melody"
        }
    }
    
    def __init__(self):
        self.current_score_name = "ode_to_joy"
        self.current_score = self.SCORES[self.current_score_name]
        self.current_index = 0
        self.total_notes = len(self.current_score["notes"])
        self.completed = False
        
    def get_next_note(self):
        """Get the next note to play"""
        if self.completed:
            return None, None
        
        note = self.current_score["notes"][self.current_index]
        duration = self.current_score["duration"]
        
        # Advance index
        self.current_index += 1
        if self.current_index >= self.total_notes:
            self.completed = True
        
        return note, duration
    
    def get_progress(self):
        """Get current progress percentage"""
        if self.total_notes == 0:
            return 0
        return (self.current_index / self.total_notes) * 100
    
    def get_current_note_display(self):
        """Get formatted current note for display"""
        if self.completed:
            return "FINISHED"
        note = self.current_score["notes"][self.current_index]
        if isinstance(note, list):
            return "+".join(note)
        return note
    
    def get_score_info(self):
        return {
            "name": self.current_score["name"],
            "type": self.current_score["type"],
            "position": self.current_index,
            "total": self.total_notes,
            "completed": self.completed
        }
    
    def change_score(self, direction=1):
        """Change to next/previous score"""
        scores_list = list(self.SCORES.keys())
        current_idx = scores_list.index(self.current_score_name)
        new_idx = (current_idx + direction) % len(scores_list)
        self.current_score_name = scores_list[new_idx]
        self.current_score = self.SCORES[self.current_score_name]
        self.current_index = 0
        self.completed = False
        return self.current_score_name
    
    def reset(self):
        self.current_index = 0
        self.completed = False

# ============================================
# KEYBOARD CONDUCTOR WITH MUSIC
# ============================================
class MusicConductor:
    def __init__(self):
        self.beat_times = deque(maxlen=10)
        self.beat_count = 0
        self.current_bpm = 0
        self.last_beat_time = 0
        self.last_beat_name = ""
        self.last_beat_display = 0
        
        # Pattern tracking
        self.beat_phase = 0
        self.expected_next = "DOWN"
        
        # Music components
        self.note_gen = NoteGenerator()
        self.score = MusicScore()
        
        # Auto-advance mode
        self.auto_advance = CONFIG["auto_advance"]
        self.auto_timer = None
        self.last_auto_time = 0
        
        # Metronome
        self.metronome_enabled = True
        
    def register_beat_and_play(self, beat_name):
        """Register beat and play the next note from sheet music"""
        now = time.time()
        
        # Cooldown check
        if now - self.last_beat_time < CONFIG["beat_cooldown"]:
            return False, "TOO FAST"
        
        # Pattern validation
        is_correct = self.validate_pattern(beat_name)
        if not is_correct:
            return False, f"WRONG (expected {self.expected_next})"
        
        # Register beat
        self.beat_count += 1
        self.beat_times.append(now)
        self.last_beat_time = now
        self.last_beat_display = now
        self.last_beat_name = beat_name
        
        # Update pattern phase
        self.update_phase(beat_name)
        
        # Calculate BPM
        if len(self.beat_times) >= 3:
            intervals = []
            for i in range(1, len(self.beat_times)):
                interval = self.beat_times[i] - self.beat_times[i-1]
                if 0.25 < interval < 1.5:
                    intervals.append(interval)
            if intervals:
                avg = sum(intervals) / len(intervals)
                self.current_bpm = int(60.0 / avg)
        
        # PLAY THE NEXT NOTE FROM SHEET MUSIC
        note, duration = self.score.get_next_note()
        
        if note:
            # Play the note based on type
            if isinstance(note, list):
                # Chord
                self.note_gen.play_chord(note, duration)
                note_display = "+".join(note)
            else:
                # Single note
                self.note_gen.play_note(note, duration)
                note_display = note
            
            # Metronome click (optional)
            if self.metronome_enabled:
                self.note_gen.play_beep()
            
            return True, note_display
        else:
            # Score completed
            return True, "COMPLETED!"
    
    def validate_pattern(self, beat_name):
        """Check if beat follows correct pattern"""
        if CONFIG["time_signature"] == "4/4":
            expected = ["DOWN", "IN", "OUT", "UP"]
        elif CONFIG["time_signature"] == "3/4":
            expected = ["DOWN", "OUT", "UP"]
        else:
            expected = ["DOWN", "UP"]
        
        self.expected_next = expected[self.beat_phase % len(expected)]
        return beat_name == self.expected_next
    
    def update_phase(self, beat_name):
        """Update current position in pattern"""
        if CONFIG["time_signature"] == "4/4":
            order = ["DOWN", "IN", "OUT", "UP"]
        elif CONFIG["time_signature"] == "3/4":
            order = ["DOWN", "OUT", "UP"]
        else:
            order = ["DOWN", "UP"]
        
        for i, name in enumerate(order):
            if beat_name == name:
                self.beat_phase = (i + 1) % len(order)
                break
        
        self.expected_next = order[self.beat_phase]
    
    def is_beat_flash(self):
        return (time.time() - self.last_beat_display) < 0.2
    
    def get_last_beat_name(self):
        return self.last_beat_name
    
    def get_stats(self):
        score_info = self.score.get_score_info()
        return {
            "beats": self.beat_count,
            "bpm": self.current_bpm,
            "time_sig": CONFIG["time_signature"],
            "next_beat": self.expected_next,
            "score_name": score_info["name"],
            "score_pos": score_info["position"],
            "score_total": score_info["total"],
            "score_completed": score_info["completed"],
            "auto_advance": self.auto_advance,
        }
    
    def change_time_signature(self):
        signatures = ["4/4", "3/4", "2/4"]
        current_idx = signatures.index(CONFIG["time_signature"])
        next_idx = (current_idx + 1) % len(signatures)
        CONFIG["time_signature"] = signatures[next_idx]
        self.beat_phase = 0
        print(f"\n[TIME SIGNATURE: {CONFIG['time_signature']}]")
        return CONFIG["time_signature"]
    
    def change_score(self, direction=1):
        new_score = self.score.change_score(direction)
        print(f"\n[SCORE: {self.score.SCORES[new_score]['name']}]")
        return new_score
    
    def toggle_metronome(self):
        self.metronome_enabled = not self.metronome_enabled
        print(f"\n[METRONOME: {'ON' if self.metronome_enabled else 'OFF'}]")
        return self.metronome_enabled
    
    def toggle_auto_advance(self):
        self.auto_advance = not self.auto_advance
        CONFIG["auto_advance"] = self.auto_advance
        print(f"\n[AUTO ADVANCE: {'ON' if self.auto_advance else 'OFF'}]")
        return self.auto_advance
    
    def reset(self):
        self.beat_count = 0
        self.current_bpm = 0
        self.beat_times.clear()
        self.last_beat_time = 0
        self.beat_phase = 0
        self.score.reset()
        print("\n[RESET - ALL COUNTERS CLEARED]")

# ============================================
# DRAWING FUNCTIONS
# ============================================
def draw_pattern_guide(frame, center_x, center_y, size=80):
    """Draw conducting pattern guide"""
    if CONFIG["time_signature"] == "4/4":
        points = [
            (center_x, center_y - size),
            (center_x, center_y),
            (center_x - size, center_y),
            (center_x + size, center_y),
            (center_x, center_y - size),
        ]
        arrows = ["", "DOWN", "LEFT", "RIGHT", "UP"]
    elif CONFIG["time_signature"] == "3/4":
        points = [
            (center_x, center_y - size),
            (center_x, center_y),
            (center_x + size, center_y),
            (center_x, center_y - size),
        ]
        arrows = ["", "DOWN", "RIGHT", "UP"]
    else:
        points = [
            (center_x, center_y - size),
            (center_x, center_y),
            (center_x, center_y - size),
        ]
        arrows = ["", "DOWN", "UP"]
    
    for i in range(len(points)-1):
        cv2.line(frame, points[i], points[i+1], (80, 80, 200), 3)
        cv2.circle(frame, points[i], 10, (50, 50, 150), -1)
        cv2.circle(frame, points[i], 10, (100, 100, 255), 2)
        cv2.putText(frame, arrows[i], (points[i][0]-25, points[i][1]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.circle(frame, points[-1], 10, (50, 50, 150), -1)
    cv2.circle(frame, points[-1], 10, (100, 100, 255), 2)
    cv2.putText(frame, arrows[-1], (points[-1][0]-25, points[-1][1]-15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame

def draw_music_sheet(frame, conductor):
    """Draw the music sheet with current position"""
    h, w = frame.shape[:2]
    stats = conductor.get_stats()
    
    # Music sheet panel
    sheet_x = w - 350
    sheet_y = 80
    
    cv2.rectangle(frame, (sheet_x, sheet_y), (w - 20, sheet_y + 250), (20, 20, 40), -1)
    cv2.rectangle(frame, (sheet_x, sheet_y), (w - 20, sheet_y + 250), (100, 100, 150), 2)
    
    cv2.putText(frame, "MUSIC SHEET", (sheet_x + 100, sheet_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    cv2.putText(frame, f"Score: {stats['score_name']}", (sheet_x + 10, sheet_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Progress bar
    if stats['score_total'] > 0:
        progress = stats['score_pos'] / stats['score_total']
        bar_width = 280
        cv2.rectangle(frame, (sheet_x + 10, sheet_y + 85), 
                     (sheet_x + 10 + bar_width, sheet_y + 105), (50, 50, 50), -1)
        cv2.rectangle(frame, (sheet_x + 10, sheet_y + 85), 
                     (sheet_x + 10 + int(bar_width * progress), sheet_y + 105), 
                     (0, 200, 0), -1)
        
        cv2.putText(frame, f"Note: {stats['score_pos']}/{stats['score_total']}", 
                   (sheet_x + 10, sheet_y + 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    # Staff lines (5 lines like real sheet music)
    staff_y_start = sheet_y + 150
    for i in range(5):
        y = staff_y_start + i * 10
        cv2.line(frame, (sheet_x + 10, y), (sheet_x + 290, y), (150, 150, 200), 1)
    
    # Current note placeholder
    cv2.putText(frame, "Next Note:", (sheet_x + 10, staff_y_start - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "-> PRESS KEY", (sheet_x + 80, staff_y_start + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame

def draw_keyboard_indicator(frame):
    """Draw keyboard layout"""
    h, w = frame.shape[:2]
    
    box_x = w - 350
    box_y = h - 170
    
    cv2.rectangle(frame, (box_x, box_y), (w - 40, h - 30), (30, 30, 50), -1)
    cv2.rectangle(frame, (box_x, box_y), (w - 40, h - 30), (100, 100, 150), 2)
    
    cv2.putText(frame, "KEYBOARD CONTROLS", (box_x + 50, box_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    # Arrow keys layout
    keys = [
        ("UP", 110, 45),
        ("LEFT", 70, 70),
        ("DOWN", 110, 70),
        ("RIGHT", 150, 70),
    ]
    
    for name, x_off, y_off in keys:
        key_x = box_x + x_off
        key_y = box_y + y_off
        cv2.rectangle(frame, (key_x, key_y), (key_x + 40, key_y + 25), (60, 60, 80), -1)
        cv2.rectangle(frame, (key_x, key_y), (key_x + 40, key_y + 25), (0, 255, 0), 2)
        cv2.putText(frame, name, (key_x + 8, key_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    return frame

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    conductor = MusicConductor()
    
    # Install simpleaudio if not present
    try:
        import simpleaudio
    except ImportError:
        print("Installing required audio library...")
        os.system("pip install simpleaudio")
        import simpleaudio
    
    cv2.namedWindow('Virtual Conductor - Real-Time Music', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Virtual Conductor - Real-Time Music', 1200, 800)
    
    h, w = 800, 1200
    
    print("=" * 70)
    print("VIRTUAL CHOIR CONDUCTOR - REAL-TIME MUSIC PLAYER")
    print("=" * 70)
    print("\n[HOW IT WORKS]")
    print("  Each correct key press plays the NEXT NOTE from the sheet music")
    print("  The conducting pattern must be followed in order")
    print("\n[CURRENT SETUP]")
    print(f"  Score: Ode to Joy")
    print(f"  Pattern: {conductor.get_stats()['time_sig']}")
    print("\n[CONTROLS]")
    print("  Arrow Keys - Conduct (play next note)")
    print("  [1] [2] [3] - Change score")
    print("  [t] - Change time signature")
    print("  [m] - Toggle metronome click")
    print("  [a] - Toggle auto-advance mode")
    print("  [r] - Reset")
    print("  [s] - Save results")
    print("  [q] - Quit")
    print("\n[PRESS ARROW KEYS TO PLAY MUSIC]")
    print("=" * 70)
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    error_message = ""
    error_time = 0
    last_played_note = ""
    note_display_time = 0
    
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Grid background
        for i in range(0, w, 50):
            cv2.line(frame, (i, 0), (i, h), (25, 25, 35), 1)
        for i in range(0, h, 50):
            cv2.line(frame, (0, i), (w, i), (25, 25, 35), 1)
        
        # Title
        cv2.putText(frame, "VIRTUAL CHOIR CONDUCTOR", (w//2 - 220, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Real-Time Music Player", (w//2 - 150, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw pattern guide
        frame = draw_pattern_guide(frame, w//2, h//2 + 50, 100)
        
        # Draw music sheet
        frame = draw_music_sheet(frame, conductor)
        
        # Draw keyboard indicator
        frame = draw_keyboard_indicator(frame)
        
        stats = conductor.get_stats()
        
        # Information panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 100), (320, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, "STATUS", (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        
        cv2.putText(frame, f"Time Sig: {stats['time_sig']}", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        cv2.putText(frame, f"Beats: {stats['beats']}", (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"BPM: {stats['bpm']}", (20, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Next: {stats['next_beat']}", (20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(frame, f"FPS: {fps}", (20, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Mode indicators
        mode_y = 310
        cv2.putText(frame, "MODES:", (20, mode_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        
        metronome_status = "ON" if conductor.metronome_enabled else "OFF"
        metronome_color = (0, 255, 0) if conductor.metronome_enabled else (100, 100, 100)
        cv2.putText(frame, f"Metronome: {metronome_status}", (20, mode_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, metronome_color, 1)
        
        auto_status = "ON" if stats['auto_advance'] else "OFF"
        auto_color = (0, 255, 0) if stats['auto_advance'] else (100, 100, 100)
        cv2.putText(frame, f"Auto-Advance: {auto_status}", (20, mode_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, auto_color, 1)
        
        # Error message
        if error_message and time.time() - error_time < 1.0:
            cv2.putText(frame, error_message, (w//2 - 150, h//2 + 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Beat flash effect
        if conductor.is_beat_flash():
            beat_name = conductor.get_last_beat_name()
            beat_display = f"[{beat_name} BEAT]"
            cv2.putText(frame, beat_display, (w//2 - 120, h//2 - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
        
        # FPS update
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Instructions
        cv2.putText(frame, "[1-3] Score  [t] TimeSig  [m] Metro  [a] Auto  [r] Reset  [s] Save  [q] Quit", 
                   (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        cv2.imshow('Virtual Conductor - Real-Time Music', frame)
        
        # Keyboard handling
        key = cv2.waitKey(50) & 0xFF
        
        # Arrow keys
        if key == 82:  # Up
            success, result = conductor.register_beat_and_play("UP")
            if not success:
                error_message = f"ERROR: {result}"
                error_time = time.time()
            else:
                last_played_note = result
                note_display_time = time.time()
                print(f"[UP] Played: {result} | Beats: {conductor.beat_count}")
        
        elif key == 84:  # Down
            success, result = conductor.register_beat_and_play("DOWN")
            if not success:
                error_message = f"ERROR: {result}"
                error_time = time.time()
            else:
                last_played_note = result
                note_display_time = time.time()
                print(f"[DOWN] Played: {result} | Beats: {conductor.beat_count}")
        
        elif key == 81:  # Left
            success, result = conductor.register_beat_and_play("IN")
            if not success:
                error_message = f"ERROR: {result}"
                error_time = time.time()
            else:
                last_played_note = result
                note_display_time = time.time()
                print(f"[LEFT] Played: {result} | Beats: {conductor.beat_count}")
        
        elif key == 83:  # Right
            success, result = conductor.register_beat_and_play("OUT")
            if not success:
                error_message = f"ERROR: {result}"
                error_time = time.time()
            else:
                last_played_note = result
                note_display_time = time.time()
                print(f"[RIGHT] Played: {result} | Beats: {conductor.beat_count}")
        
        # Control keys
        elif key == ord('q'):
            break
        elif key == ord('r'):
            conductor.reset()
            error_message = ""
        elif key == ord('t'):
            conductor.change_time_signature()
        elif key == ord('m'):
            conductor.toggle_metronome()
        elif key == ord('a'):
            conductor.toggle_auto_advance()
        elif key == ord('1'):
            conductor.change_score(-1)
        elif key == ord('2'):
            # Reset to current score
            conductor.score.reset()
            print("\n[SCORE RESET]")
        elif key == ord('3'):
            conductor.change_score(1)
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"music_conductor_results_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write("Music Conductor - Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Time signature: {stats['time_sig']}\n")
                f.write(f"Beats: {stats['beats']}\n")
                f.write(f"BPM: {stats['bpm']}\n")
                f.write(f"Score: {stats['score_name']}\n")
                f.write(f"Progress: {stats['score_pos']}/{stats['score_total']}\n")
            print(f"\n[Saved to {filename}]")
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    stats = conductor.get_stats()
    print(f"Total beats: {conductor.beat_count}")
    print(f"Final BPM: {conductor.current_bpm}")
    print(f"Score: {stats['score_name']}")
    print(f"Progress: {stats['score_pos']}/{stats['score_total']}")
    print("=" * 50)

if __name__ == "__main__":
    main()
