#!/usr/bin/env python3
"""
Virtual Choir Conductor - Score-Determined Time Signatures
Each score has its own time signature.
Select a score, and the correct pattern is automatically used.
"""

import cv2
import numpy as np
import time
import simpleaudio as sa
from collections import deque
from datetime import datetime

# ============================================
# AUDIO NOTE GENERATOR
# ============================================
class NoteGenerator:
    FREQUENCIES = {
        'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
        'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
        'G5': 783.99, 'A5': 880.00, 'B5': 987.77, 'C6': 1046.50,
    }
    
    def __init__(self):
        self.sample_rate = 44100
        self.current_audio = None
        
    def generate_tone(self, frequency, duration=0.3, volume=0.3):
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        wave = volume * np.sin(2 * np.pi * frequency * t)
        fade_samples = int(0.01 * self.sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
        return (wave * 32767).astype(np.int16)
    
    def play_note(self, note_name, duration=0.3):
        if note_name in self.FREQUENCIES:
            freq = self.FREQUENCIES[note_name]
            audio = self.generate_tone(freq, duration)
            if self.current_audio:
                self.current_audio.stop()
            self.current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)
            return True
        return False
    
    def play_beep(self):
        audio = self.generate_tone(880, 0.08, 0.15)
        self.current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)

# ============================================
# MUSIC SCORES WITH EMBEDDED TIME SIGNATURES
# ============================================
class MusicScore:
    """
    Each score has its own time signature.
    When a score is selected, the conducting pattern is determined by that signature.
    """
    
    SCORES = [
        {   # Score 1
            "id": 1,
            "name": "March in C",
            "composer": "Traditional",
            "time_signature": "2/4",
            "pattern": ["DOWN", "UP"],
            "pattern_display": "DOWN -> UP",
            "description": "March style - 2 beats per measure",
            "notes": ['C4', 'D4', 'E4', 'F4', 'G4', 'G4', 'G4',
                     'C4', 'D4', 'E4', 'F4', 'G4', 'G4', 'G4',
                     'A4', 'A4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4'],
            "duration": 0.35,
        },
        {   # Score 2
            "id": 2,
            "name": "Ode to Joy",
            "composer": "Beethoven",
            "time_signature": "4/4",
            "pattern": ["DOWN", "IN", "OUT", "UP"],
            "pattern_display": "DOWN -> IN -> OUT -> UP",
            "description": "Classical - 4 beats per measure",
            "notes": ['E4', 'E4', 'F4', 'G4', 'G4', 'F4', 'E4', 'D4',
                     'C4', 'C4', 'D4', 'E4', 'E4', 'D4', 'D4',
                     'E4', 'E4', 'F4', 'G4', 'G4', 'F4', 'E4', 'D4',
                     'C4', 'C4', 'D4', 'E4', 'D4', 'C4', 'C4'],
            "duration": 0.45,
        },
        {   # Score 3
            "id": 3,
            "name": "Waltz in C",
            "composer": "Traditional",
            "time_signature": "3/4",
            "pattern": ["DOWN", "OUT", "UP"],
            "pattern_display": "DOWN -> OUT -> UP",
            "description": "Waltz - 3 beats per measure",
            "notes": ['C4', 'E4', 'G4', 'C4', 'E4', 'G4', 
                     'D4', 'F4', 'A4', 'D4', 'F4', 'A4',
                     'E4', 'G4', 'B4', 'E4', 'G4', 'B4',
                     'F4', 'A4', 'C5', 'F4', 'A4', 'C5'],
            "duration": 0.4,
        },
        {   # Score 4
            "id": 4,
            "name": "Polka",
            "composer": "Traditional",
            "time_signature": "2/4",
            "pattern": ["DOWN", "UP"],
            "pattern_display": "DOWN -> UP",
            "description": "Polka - 2 beats per measure",
            "notes": ['C4', 'C4', 'D4', 'E4', 'C4', 'E4', 'D4', 'C4',
                     'G3', 'G3', 'A3', 'B3', 'C4', 'B3', 'A3', 'G3',
                     'C4', 'C4', 'D4', 'E4', 'C4', 'E4', 'D4', 'C4',
                     'G3', 'G3', 'A3', 'B3', 'C4', 'C4', 'C4'],
            "duration": 0.35,
        },
        {   # Score 5
            "id": 5,
            "name": "Canon in D",
            "composer": "Pachelbel",
            "time_signature": "4/4",
            "pattern": ["DOWN", "IN", "OUT", "UP"],
            "pattern_display": "DOWN -> IN -> OUT -> UP",
            "description": "Baroque - 4 beats per measure",
            "notes": ['D4', 'A3', 'B3', 'F#4', 'G4', 'D4', 'G4', 'A4',
                     'D4', 'A3', 'B3', 'F#4', 'G4', 'D4', 'G4', 'A4',
                     'B3', 'F#4', 'G4', 'D4', 'G4', 'A4', 'D5', 'C5',
                     'B4', 'A4', 'G4', 'F#4', 'G4', 'A4', 'B4', 'C5', 'D5'],
            "duration": 0.5,
        },
        {   # Score 6
            "id": 6,
            "name": "Minuet",
            "composer": "Bach",
            "time_signature": "3/4",
            "pattern": ["DOWN", "OUT", "UP"],
            "pattern_display": "DOWN -> OUT -> UP",
            "description": "Minuet - 3 beats per measure",
            "notes": ['D4', 'D4', 'E4', 'F4', 'G4', 'A4', 'G4', 'F4',
                     'E4', 'D4', 'C4', 'B3', 'A3', 'B3', 'C4', 'D4',
                     'E4', 'D4', 'C4', 'B3', 'C4', 'D4', 'E4', 'F4',
                     'G4', 'A4', 'B4', 'C5', 'D5', 'C5', 'B4', 'A4',
                     'G4', 'F4', 'E4', 'D4'],
            "duration": 0.5,
        },
        {   # Score 7
            "id": 7,
            "name": "Rock Beat",
            "composer": "Modern",
            "time_signature": "4/4",
            "pattern": ["DOWN", "IN", "OUT", "UP"],
            "pattern_display": "DOWN -> IN -> OUT -> UP",
            "description": "Rock - 4 beats per measure",
            "notes": ['C4', 'C4', 'G4', 'G4', 'A4', 'A4', 'G4',
                     'F4', 'F4', 'E4', 'E4', 'D4', 'D4', 'C4',
                     'G4', 'G4', 'F4', 'F4', 'E4', 'E4', 'D4',
                     'G4', 'G4', 'F4', 'F4', 'E4', 'E4', 'D4', 'C4'],
            "duration": 0.3,
        },
    ]
    
    def __init__(self):
        self.current_idx = 1  # Start with Ode to Joy (index 1)
        self.load_score(self.current_idx)
        
    def load_score(self, idx):
        """Load a score by index"""
        self.current_idx = idx % len(self.SCORES)
        self.current_score = self.SCORES[self.current_idx]
        self.name = self.current_score["name"]
        self.composer = self.current_score["composer"]
        self.time_signature = self.current_score["time_signature"]
        self.pattern = self.current_score["pattern"]
        self.pattern_display = self.current_score["pattern_display"]
        self.description = self.current_score["description"]
        self.notes = self.current_score["notes"]
        self.duration = self.current_score["duration"]
        self.current_note_idx = 0
        self.completed = False
        
    def get_info(self):
        return {
            "id": self.current_score["id"],
            "name": self.name,
            "composer": self.composer,
            "time_signature": self.time_signature,
            "pattern": self.pattern,
            "pattern_display": self.pattern_display,
            "description": self.description,
            "position": self.current_note_idx,
            "total": len(self.notes),
            "completed": self.completed
        }
    
    def get_next_note(self):
        if self.completed:
            return None, None
        note = self.notes[self.current_note_idx]
        self.current_note_idx += 1
        if self.current_note_idx >= len(self.notes):
            self.completed = True
        return note, self.duration
    
    def next_score(self):
        self.load_score(self.current_idx + 1)
        return self.get_info()
    
    def prev_score(self):
        self.load_score(self.current_idx - 1)
        return self.get_info()
    
    def reset(self):
        self.current_note_idx = 0
        self.completed = False

# ============================================
# CONDUCTOR (Uses score's time signature)
# ============================================
class ScoreConductor:
    def __init__(self):
        self.beat_times = deque(maxlen=10)
        self.beat_count = 0
        self.current_bpm = 0
        self.last_beat_time = 0
        self.last_beat_name = ""
        self.last_beat_display = 0
        
        self.beat_phase = 0
        self.expected_pattern = []
        self.expected_next = ""
        
        self.note_gen = NoteGenerator()
        self.score = MusicScore()
        
        self.update_pattern_from_score()
        self.metronome_enabled = True
        
    def update_pattern_from_score(self):
        """Update conducting pattern based on current score's time signature"""
        self.expected_pattern = self.score.pattern
        self.beat_phase = 0
        self.expected_next = self.expected_pattern[0] if self.expected_pattern else "DOWN"
        return self.expected_pattern
    
    def register_beat_and_play(self, beat_name):
        now = time.time()
        
        if now - self.last_beat_time < 0.2:
            return False, "TOO FAST"
        
        if beat_name != self.expected_next:
            return False, f"WRONG (expected {self.expected_next})"
        
        self.beat_count += 1
        self.beat_times.append(now)
        self.last_beat_time = now
        self.last_beat_display = now
        self.last_beat_name = beat_name
        
        self.beat_phase = (self.beat_phase + 1) % len(self.expected_pattern)
        self.expected_next = self.expected_pattern[self.beat_phase]
        
        if len(self.beat_times) >= 3:
            intervals = []
            for i in range(1, len(self.beat_times)):
                interval = self.beat_times[i] - self.beat_times[i-1]
                if 0.25 < interval < 1.5:
                    intervals.append(interval)
            if intervals:
                avg = sum(intervals) / len(intervals)
                self.current_bpm = int(60.0 / avg)
        
        note, duration = self.score.get_next_note()
        
        if note:
            self.note_gen.play_note(note, duration)
            if self.metronome_enabled:
                self.note_gen.play_beep()
            return True, note
        else:
            return True, "SCORE COMPLETE"
    
    def is_beat_flash(self):
        return (time.time() - self.last_beat_display) < 0.2
    
    def get_last_beat_name(self):
        return self.last_beat_name
    
    def get_stats(self):
        info = self.score.get_info()
        return {
            "beats": self.beat_count,
            "bpm": self.current_bpm,
            "score_id": info["id"],
            "score_name": info["name"],
            "composer": info["composer"],
            "time_signature": info["time_signature"],
            "pattern_display": info["pattern_display"],
            "next_beat": self.expected_next,
            "score_pos": info["position"],
            "score_total": info["total"],
        }
    
    def change_score(self, direction):
        if direction == 1:
            self.score.next_score()
        else:
            self.score.prev_score()
        self.update_pattern_from_score()
        print(f"\n{'='*50}")
        print(f"SCORE {self.score.current_score['id']}: {self.score.name}")
        print(f"  Time Signature: {self.score.time_signature}")
        print(f"  Pattern: {self.score.pattern_display}")
        print(f"  Description: {self.score.description}")
        print(f"{'='*50}")
    
    def toggle_metronome(self):
        self.metronome_enabled = not self.metronome_enabled
        print(f"\n[METRONOME: {'ON' if self.metronome_enabled else 'OFF'}]")
    
    def reset(self):
        self.beat_count = 0
        self.current_bpm = 0
        self.beat_times.clear()
        self.last_beat_time = 0
        self.score.reset()
        self.beat_phase = 0
        self.expected_next = self.expected_pattern[0] if self.expected_pattern else "DOWN"
        print("\n[RESET - Counter cleared, score restarted]")

# ============================================
# DRAWING FUNCTIONS
# ============================================
def draw_pattern_guide(frame, center_x, center_y, time_sig, pattern_list, next_beat):
    """Draw conducting pattern based on time signature"""
    size = 80
    
    if time_sig == "4/4":
        points = [
            (center_x, center_y - size),
            (center_x, center_y),
            (center_x - size, center_y),
            (center_x + size, center_y),
            (center_x, center_y - size),
        ]
        arrows = ["START", "DOWN", "IN", "OUT", "UP"]
    elif time_sig == "3/4":
        points = [
            (center_x, center_y - size),
            (center_x, center_y),
            (center_x + size, center_y),
            (center_x, center_y - size),
        ]
        arrows = ["START", "DOWN", "OUT", "UP"]
    else:  # 2/4
        points = [
            (center_x, center_y - size),
            (center_x, center_y),
            (center_x, center_y - size),
        ]
        arrows = ["START", "DOWN", "UP"]
    
    # Draw lines and points
    for i in range(len(points)-1):
        is_next = (i > 0 and arrows[i] == next_beat)
        color = (0, 200, 0) if is_next else (80, 80, 200)
        circle_color = (0, 150, 0) if is_next else (50, 50, 150)
        
        cv2.line(frame, points[i], points[i+1], color, 3)
        cv2.circle(frame, points[i], 10, circle_color, -1)
        cv2.circle(frame, points[i], 10, (100, 100, 255), 2)
        cv2.putText(frame, arrows[i], (points[i][0]-40, points[i][1]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.circle(frame, points[-1], 10, (50, 50, 150), -1)
    cv2.circle(frame, points[-1], 10, (100, 100, 255), 2)
    cv2.putText(frame, arrows[-1], (points[-1][0]-40, points[-1][1]-15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame

def draw_score_panel(frame, stats):
    h, w = frame.shape[:2]
    x = w - 350
    y = 80
    
    cv2.rectangle(frame, (x, y), (w - 20, y + 320), (20, 20, 40), -1)
    cv2.rectangle(frame, (x, y), (w - 20, y + 320), (100, 100, 150), 2)
    
    # Score header
    cv2.putText(frame, f"SCORE {stats['score_id']}", (x + 110, y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    
    # Score name (highlighted)
    cv2.putText(frame, stats['score_name'], (x + 70, y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, f"by {stats['composer']}", (x + 90, y + 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # TIME SIGNATURE - prominently displayed
    cv2.putText(frame, "TIME SIGNATURE", (x + 80, y + 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    # Large time signature display
    ts_color = (0, 255, 0)
    cv2.putText(frame, f"  {stats['time_signature']}", (x + 110, y + 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, ts_color, 3)
    
    # Pattern display
    cv2.putText(frame, "CONDUCTING PATTERN:", (x + 10, y + 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, stats['pattern_display'], (x + 10, y + 225), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Progress
    if stats['score_total'] > 0:
        progress = stats['score_pos'] / stats['score_total']
        bar_width = 290
        cv2.rectangle(frame, (x + 10, y + 250), (x + 10 + bar_width, y + 270), (50, 50, 50), -1)
        cv2.rectangle(frame, (x + 10, y + 250), (x + 10 + int(bar_width * progress), y + 270), (0, 200, 0), -1)
        cv2.putText(frame, f"Progress: {stats['score_pos']}/{stats['score_total']}", 
                   (x + 10, y + 295), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    return frame

def draw_keyboard_indicator(frame, next_beat):
    h, w = frame.shape[:2]
    x = w - 350
    y = h - 120
    
    cv2.rectangle(frame, (x, y), (w - 40, h - 20), (30, 30, 50), -1)
    cv2.rectangle(frame, (x, y), (w - 40, h - 20), (100, 100, 150), 2)
    cv2.putText(frame, "NEXT KEY TO PRESS:", (x + 40, y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    key_map = {"DOWN": (110, 35, "DOWN ARROW"), "IN": (70, 60, "LEFT ARROW"),
               "OUT": (150, 60, "RIGHT ARROW"), "UP": (110, 35, "UP ARROW")}
    
    if next_beat in key_map:
        x_off, y_off, key_name = key_map[next_beat]
        kx = x + x_off
        ky = y + y_off
        cv2.rectangle(frame, (kx, ky), (kx + 70, ky + 25), (0, 100, 0), -1)
        cv2.rectangle(frame, (kx, ky), (kx + 70, ky + 25), (0, 255, 0), 3)
        cv2.putText(frame, key_name, (kx + 5, ky + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    return frame

# ============================================
# MAIN
# ============================================
def main():
    conductor = ScoreConductor()
    
    cv2.namedWindow('Score Conductor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Score Conductor', 1200, 800)
    
    h, w = 800, 1200
    
    print("=" * 70)
    print("SCORE CONDUCTOR - Time Signature per Score")
    print("=" * 70)
    print("\nEach score has its own time signature.")
    print("Select a score - the correct pattern is automatically used.")
    print("\n[CURRENT SCORES]")
    for score in MusicScore.SCORES:
        print(f"  Score {score['id']}: {score['name']} - {score['time_signature']} ({score['pattern_display']})")
    print("\n[CONTROLS]")
    print("  Arrow Keys - Conduct (follow the pattern shown)")
    print("  [1] / [2] - Previous/Next score")
    print("  [m] - Metronome on/off")
    print("  [r] - Reset current score")
    print("  [s] - Save results")
    print("  [q] - Quit")
    print("=" * 70)
    
    # Show initial score info
    stats = conductor.get_stats()
    print(f"\nSTARTING WITH:")
    print(f"  Score {stats['score_id']}: {stats['score_name']}")
    print(f"  Time Signature: {stats['time_signature']}")
    print(f"  Pattern: {stats['pattern_display']}")
    print("=" * 70)
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    error_msg = ""
    error_time = 0
    
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Grid background
        for i in range(0, w, 50):
            cv2.line(frame, (i, 0), (i, h), (25, 25, 35), 1)
        for i in range(0, h, 50):
            cv2.line(frame, (0, i), (w, i), (25, 25, 35), 1)
        
        stats = conductor.get_stats()
        
        # Title
        cv2.putText(frame, "SCORE CONDUCTOR", (w//2 - 150, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw pattern guide based on current score's time signature
        frame = draw_pattern_guide(frame, w//2, h//2 + 50, 
                                   stats['time_signature'], 
                                   conductor.expected_pattern, 
                                   stats['next_beat'])
        
        frame = draw_score_panel(frame, stats)
        frame = draw_keyboard_indicator(frame, stats['next_beat'])
        
        # Info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 100), (300, 240), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, "STATUS", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame, f"Score: {stats['score_id']}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(frame, f"Time Sig: {stats['time_signature']}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Beats: {stats['beats']}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"BPM: {stats['bpm']}", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Error message
        if error_msg and time.time() - error_time < 1.0:
            cv2.putText(frame, error_msg, (w//2 - 150, h//2 + 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Beat flash
        if conductor.is_beat_flash():
            beat = conductor.get_last_beat_name()
            cv2.putText(frame, f"[{beat} BEAT]", (w//2 - 100, h//2 - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        
        # FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps}", (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "[1]Prev [2]Next [m]Metro [r]Reset [s]Save [q]Quit", 
                   (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        cv2.imshow('Score Conductor', frame)
        
        key = cv2.waitKey(50) & 0xFF
        
        # Arrow keys for conducting
        if key == 82:  # Up
            ok, result = conductor.register_beat_and_play("UP")
            print(f"[UP] {result}" if ok else f"[UP] ERROR: {result}")
            if not ok:
                error_msg = result
                error_time = time.time()
        elif key == 84:  # Down
            ok, result = conductor.register_beat_and_play("DOWN")
            print(f"[DOWN] {result}" if ok else f"[DOWN] ERROR: {result}")
            if not ok:
                error_msg = result
                error_time = time.time()
        elif key == 81:  # Left
            ok, result = conductor.register_beat_and_play("IN")
            print(f"[IN] {result}" if ok else f"[IN] ERROR: {result}")
            if not ok:
                error_msg = result
                error_time = time.time()
        elif key == 83:  # Right
            ok, result = conductor.register_beat_and_play("OUT")
            print(f"[OUT] {result}" if ok else f"[OUT] ERROR: {result}")
            if not ok:
                error_msg = result
                error_time = time.time()
        elif key == ord('q'):
            break
        elif key == ord('r'):
            conductor.reset()
            error_msg = ""
        elif key == ord('1'):
            conductor.change_score(-1)
        elif key == ord('2'):
            conductor.change_score(1)
        elif key == ord('m'):
            conductor.toggle_metronome()
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"score_conductor_results_{ts}.txt", 'w') as f:
                f.write(f"Score: {stats['score_id']} - {stats['score_name']}\n")
                f.write(f"Time Signature: {stats['time_signature']}\n")
                f.write(f"Pattern: {stats['pattern_display']}\n")
                f.write(f"Beats: {stats['beats']}\n")
                f.write(f"BPM: {stats['bpm']}\n")
                f.write(f"Progress: {stats['score_pos']}/{stats['score_total']}\n")
            print(f"\n[Saved to score_conductor_results_{ts}.txt]")
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    stats = conductor.get_stats()
    print(f"Final Score: {stats['score_id']} - {stats['score_name']}")
    print(f"Time Signature: {stats['time_signature']}")
    print(f"Total Beats: {conductor.beat_count}")
    print(f"Final BPM: {conductor.current_bpm}")
    print("=" * 50)

if __name__ == "__main__":
    main()
