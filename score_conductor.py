#!/usr/bin/env python3
"""
Virtual Choir Conductor - Score-Determined Time Signatures
With Musical Score Display, Correct Beat Numbering, and Completion Banner After Last Note
"""

import cv2
import numpy as np
import time
import simpleaudio as sa
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import threading


# ============================================
# CONSTANTS AND ENUMS
# ============================================

class BeatDirection(Enum):
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"

    @classmethod
    def from_keycode(cls, keycode: int) -> Optional['BeatDirection']:
        mapping = {
            84: cls.DOWN,
            82: cls.UP,
            81: cls.LEFT,
            83: cls.RIGHT
        }
        return mapping.get(keycode)

    def to_display_name(self) -> str:
        names = {
            "DOWN": "DOWN ARROW",
            "LEFT": "LEFT ARROW",
            "RIGHT": "RIGHT ARROW",
            "UP": "UP ARROW"
        }
        return names[self.value]


@dataclass
class VisualConstants:
    WINDOW_WIDTH: int = 1400
    WINDOW_HEIGHT: int = 900
    PATTERN_SIZE: int = 70
    SCORE_PANEL_WIDTH: int = 320
    SCORE_PANEL_HEIGHT: int = 400
    SCORE_PANEL_X_OFFSET: int = 20
    SCORE_PANEL_Y_OFFSET: int = 80
    KEYBOARD_PANEL_HEIGHT: int = 100
    INFO_PANEL_WIDTH: int = 285
    INFO_PANEL_HEIGHT: int = 240
    NOTATION_WIDTH: int = 500
    NOTATION_HEIGHT: int = 200


# ============================================
# MUSICAL NOTE WITH ARTICULATION
# ============================================

@dataclass
class MusicalEvent:
    duration: float
    note: Optional[str] = None
    articulation: str = "normal"
    is_rest: bool = False
    beat_position: int = 0

    @classmethod
    def note(cls, note_name: str, duration: float, articulation: str = "normal", beat_pos: int = 0):
        return cls(duration=duration, note=note_name, articulation=articulation, is_rest=False, beat_position=beat_pos)

    @classmethod
    def rest(cls, duration: float, beat_pos: int = 0):
        return cls(duration=duration, note=None, articulation="silence", is_rest=True, beat_position=beat_pos)


# ============================================
# AUDIO MODULE
# ============================================

class AudioGenerator:
    FREQUENCIES: Dict[str, float] = {
        'C2': 65.41, 'D2': 73.42, 'E2': 82.41, 'F2': 87.31, 'G2': 98.00, 'A2': 110.00, 'B2': 123.47,
        'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        'C6': 1046.50, 'D6': 1174.66, 'E6': 1318.51, 'F6': 1396.91, 'G6': 1567.98, 'A6': 1760.00,
        'F#4': 369.99,
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._current_audio = None
        self._last_audio_duration = 0
        self._last_audio_start_time = 0

    def generate_note_with_articulation(self, frequency: float, duration: float,
                                        articulation: str = "normal", volume: float = 0.3) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)

        if articulation == "staccato":
            actual_duration = duration * 0.5
            envelope = np.ones_like(t)
            envelope[t > actual_duration] = np.linspace(1, 0, len(t[t > actual_duration]))
            wave = wave * envelope * volume
        elif articulation == "tenuto":
            envelope = 0.8 + 0.2 * np.sin(np.pi * t / duration)
            wave = wave * envelope * volume
        elif articulation == "accent":
            attack = np.exp(-t * 15)
            envelope = attack * 1.5
            envelope = np.clip(envelope, 0, 1)
            wave = wave * envelope * volume
        else:
            fade_samples = int(0.008 * self.sample_rate)
            if len(wave) > 2 * fade_samples:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                wave[:fade_samples] *= fade_in
                wave[-fade_samples:] *= fade_out
            wave = wave * volume

        wave = np.clip(wave, -0.99, 0.99)
        return (wave * 32767).astype(np.int16)

    def play_musical_event(self, event: MusicalEvent) -> Tuple[bool, float]:
        """Returns (success, duration)"""
        if event.is_rest:
            silence = np.zeros(int(self.sample_rate * event.duration), dtype=np.int16)
            if self._current_audio:
                self._current_audio.stop()
            self._current_audio = sa.play_buffer(silence, 1, 2, self.sample_rate)
            self._last_audio_start_time = time.time()
            self._last_audio_duration = event.duration
            return True, event.duration

        if event.note not in self.FREQUENCIES:
            return False, 0

        freq = self.FREQUENCIES[event.note]
        audio = self.generate_note_with_articulation(freq, event.duration, event.articulation)

        if self._current_audio:
            self._current_audio.stop()

        self._current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)
        self._last_audio_start_time = time.time()
        self._last_audio_duration = event.duration

        return True, event.duration

    def play_metronome_click(self, is_downbeat: bool = False) -> None:
        duration = 0.05
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        freq = 880 if not is_downbeat else 660
        volume = 0.15 if not is_downbeat else 0.25

        wave = volume * np.sin(2 * np.pi * freq * t)
        decay = np.exp(-t * 30)
        wave = wave * decay
        audio = (wave * 32767).astype(np.int16)
        sa.play_buffer(audio, 1, 2, self.sample_rate)

    def play_completion_fanfare(self) -> None:
        """Play a short fanfare when a score is completed."""

        def play():
            notes = [('C5', 0.15), ('E5', 0.15), ('G5', 0.15), ('C6', 0.3)]
            for note, dur in notes:
                if note in self.FREQUENCIES:
                    freq = self.FREQUENCIES[note]
                    t = np.linspace(0, dur, int(self.sample_rate * dur))
                    wave = 0.4 * np.sin(2 * np.pi * freq * t)
                    fade_samples = int(0.005 * self.sample_rate)
                    if len(wave) > 2 * fade_samples:
                        fade_in = np.linspace(0, 1, fade_samples)
                        fade_out = np.linspace(1, 0, fade_samples)
                        wave[:fade_samples] *= fade_in
                        wave[-fade_samples:] *= fade_out
                    audio = (wave * 32767).astype(np.int16)
                    sa.play_buffer(audio, 1, 2, self.sample_rate)
                    time.sleep(dur)

        threading.Thread(target=play, daemon=True).start()


# ============================================
# MUSICAL NOTATION DISPLAY
# ============================================

class MusicNotationDisplay:
    @staticmethod
    def create_notation_text(score_name: str, time_signature: str,
                             current_measure: int, current_beat: int,
                             beats_per_measure: int, notes_in_measure: List[Tuple[str, int]],
                             is_completed: bool = False) -> np.ndarray:
        h, w = 180, 500
        notation = np.zeros((h, w, 3), dtype=np.uint8)

        if is_completed:
            notation[:] = (200, 230, 200)
            cv2.putText(notation, "SCORE COMPLETED!", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
            return notation

        # Cream paper background
        notation[:] = (245, 245, 220)

        # Draw 5 staff lines
        staff_y_start = 50
        staff_y_end = 130
        line_spacing = 20

        for i in range(5):
            y = staff_y_start + i * line_spacing
            cv2.line(notation, (20, y), (w - 20, y), (0, 0, 0), 1)

        # Treble clef symbol (simplified)
        cv2.putText(notation, "G", (30, staff_y_start + 2 * line_spacing + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Time signature
        cv2.putText(notation, time_signature, (70, staff_y_start + 2 * line_spacing + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Beat indicator
        beat_text = f"Beat: {current_beat}/{beats_per_measure}"
        cv2.putText(notation, beat_text, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 200), 1)

        # Measure number
        cv2.putText(notation, f"Measure {current_measure}", (w - 120, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Note positions on staff (Y coordinates)
        note_positions = {
            'C4': staff_y_start + 4 * line_spacing,  # Below bottom line
            'D4': staff_y_start + 3 * line_spacing,  # Bottom line
            'E4': staff_y_start + 2 * line_spacing,  # First space
            'F4': staff_y_start + 1 * line_spacing,  # Second line
            'G4': staff_y_start + 0 * line_spacing,  # Second space
            'A4': staff_y_start - 1 * line_spacing,  # Third line
            'B4': staff_y_start - 2 * line_spacing,  # Third space
            'C5': staff_y_start - 3 * line_spacing,  # Fourth line
            'D5': staff_y_start - 4 * line_spacing,  # Fourth space
            'E5': staff_y_start - 5 * line_spacing,  # Fifth line
        }

        x_pos = 120
        note_spacing = 35

        # Draw notes on staff
        for i, (note, beat) in enumerate(notes_in_measure[:8]):
            # Draw stem connection between notes
            if i > 0:
                cv2.line(notation, (x_pos - note_spacing // 2, staff_y_start + 2 * line_spacing),
                         (x_pos, staff_y_start + 2 * line_spacing), (0, 0, 0), 1)

            if note == "REST":
                # Draw rest symbol (simplified)
                rest_y = staff_y_start + 2 * line_spacing
                cv2.rectangle(notation, (x_pos - 6, rest_y - 8), (x_pos + 2, rest_y + 4), (0, 0, 0), -1)
                cv2.putText(notation, "R", (x_pos - 6, rest_y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            elif note in note_positions:
                y_pos = note_positions[note]
                is_current = (i == 0)

                # Note head (filled circle)
                color = (0, 200, 0) if is_current else (0, 0, 0)
                thickness = -1 if is_current else 1
                cv2.circle(notation, (x_pos, y_pos), 7, color, thickness)

                # Note stem
                cv2.line(notation, (x_pos + 6, y_pos), (x_pos + 6, y_pos - 20), (0, 0, 0), 1)

                # Highlight current note with glow
                if is_current:
                    cv2.circle(notation, (x_pos, y_pos), 10, (0, 255, 0), 2)

            # Beat number below the staff
            cv2.putText(notation, str(beat), (x_pos - 5, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 100, 0), 1)

            x_pos += note_spacing

        # Bar lines
        cv2.line(notation, (110, staff_y_start - 10), (110, staff_y_end + 10), (0, 0, 0), 2)
        cv2.line(notation, (w - 20, staff_y_start - 10), (w - 20, staff_y_end + 10), (0, 0, 0), 2)

        return notation

    @staticmethod
    def create_score_info_display(score_name: str, composer: str,
                                  time_signature: str, description: str,
                                  is_completed: bool = False) -> np.ndarray:
        h, w = 200, 500
        info_panel = np.zeros((h, w, 3), dtype=np.uint8)

        if is_completed:
            info_panel[:] = (30, 60, 30)
            cv2.putText(info_panel, "COMPLETED!", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(info_panel, "Great job, Conductor!", (w // 2 - 90, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return info_panel

        info_panel[:] = (30, 30, 50)

        cv2.putText(info_panel, "SCORE INFORMATION", (w // 2 - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(info_panel, score_name, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(info_panel, f"by {composer}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(info_panel, f"Time: {time_signature}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Wrap description text
        words = description.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 < 50:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        y = 130
        for line in lines[:3]:
            cv2.putText(info_panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 150), 1)
            y += 20

        return info_panel


# ============================================
# MUSIC SCORES
# ============================================

@dataclass
class ScoreData:
    id: int
    name: str
    composer: str
    time_signature: str
    beats_per_measure: int
    pattern: List[str]
    pattern_display: str
    description: str
    musical_events: List[MusicalEvent]
    tempo_bpm: int
    time_signature_note: str


class MusicScoreLibrary:

    @staticmethod
    def _create_score_4_ode_to_joy() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 4:
                current_beat = 1

        # Main theme
        main_theme = [
            ('E4', 0.45, "normal"), ('E4', 0.45, "normal"), ('F4', 0.45, "normal"), ('G4', 0.45, "normal"),
            ('G4', 0.45, "normal"), ('F4', 0.45, "normal"), ('E4', 0.45, "normal"), ('D4', 0.45, "normal"),
            ('C4', 0.45, "normal"), ('C4', 0.45, "normal"), ('D4', 0.45, "normal"), ('E4', 0.45, "normal"),
            ('E4', 0.45, "tenuto"), ('D4', 0.45, "staccato"),
        ]

        for note, dur, art in main_theme:
            add_note(note, dur, art)

        events.append(MusicalEvent.note('D4', 0.9, "tenuto", 3))

        return ScoreData(
            id=4, name="Ode to Joy", composer="L.v. Beethoven",
            time_signature="4/4", beats_per_measure=4,
            pattern=["DOWN", "LEFT", "RIGHT", "UP"], pattern_display="DOWN -> LEFT -> RIGHT -> UP",
            description="Beethoven Symphony No. 9 - Ode to Joy theme",
            musical_events=events, tempo_bpm=100,
            time_signature_note="4/4 (Common Time)"
        )

    @staticmethod
    def _create_score_5_canon() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 4:
                current_beat = 1

        canon_notes = [
            ('D4', 0.5, "tenuto"), ('A3', 0.5, "normal"), ('B3', 0.5, "normal"), ('F#4', 0.5, "normal"),
            ('G4', 0.5, "normal"), ('D4', 0.5, "normal"), ('G4', 0.5, "normal"), ('A4', 0.5, "normal"),
            ('D4', 0.5, "accent"), ('A3', 0.5, "normal"), ('B3', 0.5, "normal"), ('F#4', 0.5, "normal"),
            ('G4', 0.5, "normal"), ('D4', 0.5, "normal"), ('G4', 0.5, "normal"), ('A4', 0.5, "normal"),
        ]

        for note, dur, art in canon_notes:
            add_note(note, dur, art)

        events.append(MusicalEvent.note('D5', 1.0, "tenuto", 2))

        return ScoreData(
            id=5, name="Canon in D", composer="J. Pachelbel",
            time_signature="4/4", beats_per_measure=4,
            pattern=["DOWN", "LEFT", "RIGHT", "UP"], pattern_display="DOWN -> LEFT -> RIGHT -> UP",
            description="Pachelbel's Canon - Famous ground bass",
            musical_events=events, tempo_bpm=80,
            time_signature_note="4/4 - Baroque canon"
        )

    @staticmethod
    def _create_score_1_march() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 2:
                current_beat = 1

        march_notes = [
            ('C4', 0.35, "staccato"), ('D4', 0.35, "staccato"), ('E4', 0.35, "staccato"),
            ('F4', 0.35, "accent"), ('G4', 0.35, "staccato"), ('G4', 0.35, "tenuto"),
            ('G4', 0.35, "staccato"), ('C4', 0.35, "accent"),
        ]

        for note, dur, art in march_notes:
            add_note(note, dur, art)

        return ScoreData(
            id=1, name="March in C", composer="Traditional",
            time_signature="2/4", beats_per_measure=2,
            pattern=["DOWN", "UP"], pattern_display="DOWN -> UP",
            description="Military march - 2 beats per measure",
            musical_events=events, tempo_bpm=120,
            time_signature_note="2/4 - Beats 1 and 2"
        )

    @staticmethod
    def _create_score_2_waltz() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 3:
                current_beat = 1

        waltz_notes = [
            ('C4', 0.4, "accent"), ('E4', 0.4, "normal"), ('G4', 0.4, "normal"),
            ('C4', 0.4, "normal"), ('E4', 0.4, "normal"), ('G4', 0.4, "normal"),
            ('D4', 0.4, "accent"), ('F4', 0.4, "normal"), ('A4', 0.4, "normal"),
        ]

        for note, dur, art in waltz_notes:
            add_note(note, dur, art)

        events.append(MusicalEvent.note('C5', 0.8, "tenuto", 3))

        return ScoreData(
            id=2, name="Waltz in C", composer="Traditional",
            time_signature="3/4", beats_per_measure=3,
            pattern=["DOWN", "RIGHT", "UP"], pattern_display="DOWN -> RIGHT -> UP",
            description="Viennese waltz - 3 beats per measure",
            musical_events=events, tempo_bpm=160,
            time_signature_note="3/4 - Waltz time"
        )

    @staticmethod
    def _create_score_3_minuet() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 3:
                current_beat = 1

        minuet_notes = [
            ('D4', 0.5, "normal"), ('D4', 0.5, "staccato"), ('E4', 0.5, "normal"),
            ('F4', 0.5, "normal"), ('G4', 0.5, "normal"), ('A4', 0.5, "normal"),
        ]

        for note, dur, art in minuet_notes:
            add_note(note, dur, art)

        events.append(MusicalEvent.note('G4', 1.0, "tenuto", 3))

        return ScoreData(
            id=3, name="Minuet", composer="J.S. Bach",
            time_signature="3/4", beats_per_measure=3,
            pattern=["DOWN", "RIGHT", "UP"], pattern_display="DOWN -> RIGHT -> UP",
            description="Bach Minuet - 3 beats per measure",
            musical_events=events, tempo_bpm=100,
            time_signature_note="3/4 - Baroque minuet"
        )

    @staticmethod
    def _create_score_6_rock() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 4:
                current_beat = 1

        rock_notes = [
            ('C4', 0.3, "accent"), ('C4', 0.3, "staccato"), ('G4', 0.3, "normal"), ('G4', 0.3, "staccato"),
            ('A4', 0.3, "accent"), ('A4', 0.3, "normal"),
        ]

        for note, dur, art in rock_notes:
            add_note(note, dur, art)

        events.append(MusicalEvent.note('G4', 0.6, "tenuto", 3))

        return ScoreData(
            id=6, name="Rock Beat", composer="Modern",
            time_signature="4/4", beats_per_measure=4,
            pattern=["DOWN", "LEFT", "RIGHT", "UP"], pattern_display="DOWN -> LEFT -> RIGHT -> UP",
            description="Rock rhythm - 4/4 time",
            musical_events=events, tempo_bpm=120,
            time_signature_note="4/4 - Rock time"
        )

    @staticmethod
    def _create_score_7_polka() -> ScoreData:
        events = []
        current_beat = 1

        def add_note(note_name, duration, articulation):
            nonlocal current_beat
            events.append(MusicalEvent.note(note_name, duration, articulation, current_beat))
            current_beat += 1
            if current_beat > 2:
                current_beat = 1

        polka_notes = [
            ('C4', 0.35, "staccato"), ('C4', 0.35, "staccato"), ('D4', 0.35, "staccato"),
            ('E4', 0.35, "staccato"), ('C4', 0.35, "accent"),
        ]

        for note, dur, art in polka_notes:
            add_note(note, dur, art)

        events.append(MusicalEvent.note('C4', 0.70, "tenuto", 2))

        return ScoreData(
            id=7, name="Polka", composer="Traditional",
            time_signature="2/4", beats_per_measure=2,
            pattern=["DOWN", "UP"], pattern_display="DOWN -> UP",
            description="Lively polka - 2/4 time",
            musical_events=events, tempo_bpm=130,
            time_signature_note="2/4 - Polka time"
        )

    @classmethod
    def get_all(cls) -> List[ScoreData]:
        return [
            cls._create_score_1_march(),
            cls._create_score_2_waltz(),
            cls._create_score_3_minuet(),
            cls._create_score_4_ode_to_joy(),
            cls._create_score_5_canon(),
            cls._create_score_6_rock(),
            cls._create_score_7_polka(),
        ]

    @classmethod
    def get_count(cls) -> int:
        return len(cls.get_all())


# ============================================
# MUSIC SCORE PLAYER
# ============================================

class MusicScorePlayer:
    def __init__(self, initial_id: int = 4):
        self._scores = MusicScoreLibrary.get_all()
        self._current_index = self._find_index_by_id(initial_id)
        self._event_index: int = 0
        self._completed: bool = False
        self._last_event_played: bool = False
        self._last_event_duration: float = 0
        self._load_current_score()

    def _find_index_by_id(self, score_id: int) -> int:
        for i, score in enumerate(self._scores):
            if score.id == score_id:
                return i
        return 0

    def _load_current_score(self) -> None:
        self.current_score = self._scores[self._current_index]
        self._event_index = 0
        self._completed = False
        self._last_event_played = False
        self._last_event_duration = 0

    @property
    def info(self) -> Dict:
        return {
            "id": self.current_score.id,
            "name": self.current_score.name,
            "composer": self.current_score.composer,
            "time_signature": self.current_score.time_signature,
            "beats_per_measure": self.current_score.beats_per_measure,
            "pattern": self.current_score.pattern,
            "pattern_display": self.current_score.pattern_display,
            "description": self.current_score.description,
            "tempo_bpm": self.current_score.tempo_bpm,
            "time_signature_note": self.current_score.time_signature_note,
            "position": self._event_index,
            "total": len(self.current_score.musical_events),
            "completed": self._completed
        }

    @property
    def pattern(self) -> List[str]:
        return self.current_score.pattern

    @property
    def time_signature(self) -> str:
        return self.current_score.time_signature

    @property
    def beats_per_measure(self) -> int:
        return self.current_score.beats_per_measure

    @property
    def suggested_tempo(self) -> int:
        return self.current_score.tempo_bpm

    @property
    def is_completed(self) -> bool:
        return self._completed

    @property
    def last_event_just_played(self) -> bool:
        return self._last_event_played

    @property
    def last_event_duration(self) -> float:
        return self._last_event_duration

    def reset_last_event_flag(self) -> None:
        self._last_event_played = False

    def get_current_notation_events(self, num_events: int = 8) -> List[Tuple[str, int]]:
        events = []
        start_idx = self._event_index
        for i in range(start_idx, min(start_idx + num_events, len(self.current_score.musical_events))):
            event = self.current_score.musical_events[i]
            if not event.is_rest and event.note:
                events.append((event.note, event.beat_position))
            elif event.is_rest:
                events.append(("REST", event.beat_position))
        return events

    def get_current_measure(self) -> int:
        if self._event_index == 0:
            return 1
        beats_so_far = 0
        for i in range(self._event_index):
            beats_so_far += 1
        return (beats_so_far // self.current_score.beats_per_measure) + 1

    def get_current_beat_in_measure(self) -> int:
        if self._event_index == 0:
            return 1
        beats_so_far = 0
        for i in range(self._event_index):
            beats_so_far += 1
        beat = (beats_so_far % self.current_score.beats_per_measure)
        return beat if beat != 0 else self.current_score.beats_per_measure

    def get_next_event(self) -> Tuple[Optional[MusicalEvent], Optional[float], bool]:
        """Returns (event, duration, is_last_event)"""
        if self._completed:
            return None, None, False

        is_last = (self._event_index == len(self.current_score.musical_events) - 1)
        event = self.current_score.musical_events[self._event_index]
        duration = event.duration
        self._event_index += 1

        if self._event_index >= len(self.current_score.musical_events):
            self._completed = True
            self._last_event_played = True
            self._last_event_duration = duration

        return event, duration, is_last

    def next(self) -> Dict:
        self._current_index = (self._current_index + 1) % len(self._scores)
        self._load_current_score()
        return self.info

    def previous(self) -> Dict:
        self._current_index = (self._current_index - 1) % len(self._scores)
        self._load_current_score()
        return self.info

    def reset(self) -> None:
        self._event_index = 0
        self._completed = False
        self._last_event_played = False
        self._last_event_duration = 0


# ============================================
# CONDUCTOR MODULE
# ============================================

class Conductor:
    MIN_BEAT_INTERVAL: float = 0.2
    MAX_BEAT_INTERVAL: float = 2.0
    MIN_BPM_INTERVAL: float = 0.25
    MAX_BPM_INTERVAL: float = 1.5
    BEAT_HISTORY_SIZE: int = 10

    def __init__(self, metronome_enabled: bool = True):
        self._beat_times = deque(maxlen=self.BEAT_HISTORY_SIZE)
        self._beat_count: int = 0
        self._measure_count: int = 0
        self._current_bpm: int = 0
        self._last_beat_time: float = 0
        self._last_beat_name: str = ""
        self._last_beat_display: float = 0
        self._beat_phase: int = 0
        self._expected_pattern: List[str] = []
        self._expected_next: str = ""
        self._completion_banner_visible: bool = False
        self._completion_banner_start_time: float = 0

        self._score_player = MusicScorePlayer()
        self._audio = AudioGenerator()
        self.metronome_enabled = metronome_enabled

        self._update_pattern()

    def _update_pattern(self) -> None:
        self._expected_pattern = self._score_player.pattern
        self._beat_phase = 0
        self._expected_next = self._expected_pattern[0] if self._expected_pattern else "DOWN"

    def _update_bpm(self) -> None:
        if len(self._beat_times) < 2:
            return

        beats_to_use = min(5, len(self._beat_times))
        intervals = []

        for i in range(len(self._beat_times) - beats_to_use + 1, len(self._beat_times)):
            interval = self._beat_times[i] - self._beat_times[i - 1]
            if self.MIN_BPM_INTERVAL < interval < self.MAX_BPM_INTERVAL:
                intervals.append(interval)

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            self._current_bpm = int(60.0 / avg_interval)

    def _check_tempo(self, now: float) -> Tuple[bool, str]:
        if self._last_beat_time > 0:
            interval = now - self._last_beat_time
            if interval < self.MIN_BEAT_INTERVAL:
                return False, f"TOO FAST (BPM > {int(60 / self.MIN_BEAT_INTERVAL)})"
            if interval > self.MAX_BEAT_INTERVAL and self._beat_count > 0:
                return False, f"TOO SLOW (BPM < {int(60 / self.MAX_BEAT_INTERVAL)})"
        return True, ""

    def _show_completion_banner(self) -> None:
        """Show completion banner after audio finishes."""
        self._completion_banner_visible = True
        self._completion_banner_start_time = time.time()
        self._audio.play_completion_fanfare()

    def register_beat(self, direction: BeatDirection) -> Tuple[bool, str]:
        # Don't accept beats if completion banner is showing
        if self._completion_banner_visible:
            return False, "SCORE COMPLETE! Press [1]/[2] for new score or [r] to replay"

        now = time.time()
        beat_name = direction.value

        tempo_ok, tempo_msg = self._check_tempo(now)
        if not tempo_ok:
            return False, tempo_msg

        if beat_name != self._expected_next:
            expected_name = self._expected_next
            return False, f"WRONG PATTERN (expected {expected_name})"

        self._beat_count += 1
        self._beat_times.append(now)
        self._last_beat_time = now
        self._last_beat_display = now
        self._last_beat_name = beat_name

        self._beat_phase = (self._beat_phase + 1) % len(self._expected_pattern)
        self._expected_next = self._expected_pattern[self._beat_phase]

        if self._beat_phase == 0:
            self._measure_count += 1

        self._update_bpm()

        event, duration, is_last = self._score_player.get_next_event()
        if event:
            success, event_duration = self._audio.play_musical_event(event)
            if self.metronome_enabled:
                is_downbeat = (event.beat_position == 1)
                self._audio.play_metronome_click(is_downbeat)

            if event.is_rest:
                msg = f"[OK] Beat {event.beat_position} -- REST"
            else:
                msg = f"[OK] Beat {event.beat_position} -- {event.note}"

            # If this was the last event, schedule completion banner after audio finishes
            if is_last:
                def show_banner_after_delay():
                    time.sleep(event_duration + 0.1)
                    self._show_completion_banner()

                threading.Thread(target=show_banner_after_delay, daemon=True).start()
                return True, msg + " (Last note - completing soon!)"

            return True, msg

        return True, "SCORE COMPLETE"

    @property
    def is_beat_flash(self) -> bool:
        return (time.time() - self._last_beat_display) < 0.15

    @property
    def last_beat_name(self) -> str:
        return self._last_beat_name

    @property
    def current_beat_number(self) -> int:
        if self._beat_phase == 0:
            return len(self._expected_pattern)
        return self._beat_phase

    @property
    def is_completed(self) -> bool:
        return self._score_player.is_completed

    @property
    def show_completion_banner(self) -> bool:
        """Show banner for 4 seconds after completion."""
        if not self._completion_banner_visible:
            return False
        if time.time() - self._completion_banner_start_time > 4.0:
            self._completion_banner_visible = False
            return False
        return True

    @property
    def stats(self) -> Dict:
        info = self._score_player.info
        return {
            "beats": self._beat_count,
            "measures": self._measure_count,
            "bpm": self._current_bpm,
            "suggested_tempo": info["tempo_bpm"],
            "score_id": info["id"],
            "score_name": info["name"],
            "composer": info["composer"],
            "time_signature": info["time_signature"],
            "beats_per_measure": info["beats_per_measure"],
            "pattern_display": info["pattern_display"],
            "description": info["description"],
            "time_signature_note": info["time_signature_note"],
            "next_beat": self._expected_next,
            "current_beat": self.current_beat_number,
            "score_pos": info["position"],
            "score_total": info["total"],
            "current_measure": self._score_player.get_current_measure(),
            "notation_events": self._score_player.get_current_notation_events(8),
            "is_completed": self.is_completed,
            "show_completion_banner": self.show_completion_banner,
        }

    @property
    def expected_pattern(self) -> List[str]:
        return self._expected_pattern

    def next_score(self) -> None:
        self._score_player.next()
        self._update_pattern()
        self._beat_count = 0
        self._measure_count = 0
        self._beat_times.clear()
        self._current_bpm = 0
        self._completion_banner_visible = False
        self._print_score_info()

    def previous_score(self) -> None:
        self._score_player.previous()
        self._update_pattern()
        self._beat_count = 0
        self._measure_count = 0
        self._beat_times.clear()
        self._current_bpm = 0
        self._completion_banner_visible = False
        self._print_score_info()

    def toggle_metronome(self) -> None:
        self.metronome_enabled = not self.metronome_enabled
        print(f"\n[METRONOME: {'ON' if self.metronome_enabled else 'OFF'}]")

    def reset(self) -> None:
        self._beat_count = 0
        self._measure_count = 0
        self._current_bpm = 0
        self._beat_times.clear()
        self._last_beat_time = 0
        self._score_player.reset()
        self._beat_phase = 0
        self._expected_next = self._expected_pattern[0] if self._expected_pattern else "DOWN"
        self._completion_banner_visible = False
        print("\n[RESET - Score restarted from beginning]")

    def _print_score_info(self) -> None:
        info = self._score_player.info
        print(f"\n{'=' * 60}")
        print(f"SCORE {info['id']}: {info['name']} (by {info['composer']})")
        print(f"{'=' * 60}")
        print(f"  Time Signature: {info['time_signature']} - {info['time_signature_note']}")
        print(f"  Suggested Tempo: {info['tempo_bpm']} BPM")
        print(f"  Conducting Pattern: {info['pattern_display']}")
        print(f"  {info['description']}")
        print(f"  Total Events: {info['total']}")
        print(f"{'=' * 60}")

    def get_final_stats(self) -> Dict:
        info = self._score_player.info
        return {
            "score_id": info["id"],
            "score_name": info["name"],
            "composer": info["composer"],
            "time_signature": info["time_signature"],
            "beats_per_measure": info["beats_per_measure"],
            "total_beats": self._beat_count,
            "total_measures": self._measure_count,
            "final_bpm": self._current_bpm,
            "suggested_tempo": info["tempo_bpm"],
        }


# ============================================
# VISUALIZER
# ============================================

class Visualizer:
    def __init__(self, constants: VisualConstants = VisualConstants()):
        self.const = constants
        self.notation_display = MusicNotationDisplay()

    def create_background(self) -> np.ndarray:
        frame = np.zeros((self.const.WINDOW_HEIGHT, self.const.WINDOW_WIDTH, 3), dtype=np.uint8)
        for x in range(0, self.const.WINDOW_WIDTH, 50):
            cv2.line(frame, (x, 0), (x, self.const.WINDOW_HEIGHT), (20, 20, 30), 1)
        for y in range(0, self.const.WINDOW_HEIGHT, 50):
            cv2.line(frame, (0, y), (self.const.WINDOW_WIDTH, y), (20, 20, 30), 1)
        return frame

    def draw_title(self, frame: np.ndarray) -> None:
        center_x = self.const.WINDOW_WIDTH // 2
        cv2.putText(frame, "VIRTUAL CHOIR CONDUCTOR", (center_x - 180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    def draw_pattern_guide(self, frame: np.ndarray, time_signature: str,
                           pattern: List[str], next_beat: str,
                           current_beat: int, show_completion: bool = False) -> np.ndarray:
        center_x = 350
        center_y = self.const.WINDOW_HEIGHT // 2 + 50
        size = self.const.PATTERN_SIZE

        if show_completion:
            cv2.putText(frame, "COMPLETED!", (center_x - 60, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            return frame

        if "4/4" in time_signature:
            points = [
                (center_x, center_y - size),
                (center_x, center_y),
                (center_x - size, center_y),
                (center_x + size, center_y),
                (center_x, center_y - size),
            ]
            beat_labels = ["START", "1 DOWN", "2 LEFT", "3 RIGHT", "4 UP"]
        elif "3/4" in time_signature:
            points = [
                (center_x, center_y - size),
                (center_x, center_y),
                (center_x + size, center_y),
                (center_x, center_y - size),
            ]
            beat_labels = ["START", "1 DOWN", "2 RIGHT", "3 UP"]
        else:
            points = [
                (center_x, center_y - size),
                (center_x, center_y),
                (center_x, center_y - size),
            ]
            beat_labels = ["START", "1 DOWN", "2 UP"]

        for i in range(len(points) - 1):
            next_beat_label = beat_labels[i + 1] if i + 1 < len(beat_labels) else ""
            beat_num = next_beat_label.split()[0] if next_beat_label else ""
            is_next = str(current_beat + 1) == beat_num
            color = (0, 200, 0) if is_next else (80, 80, 200)
            thickness = 4 if is_next else 3
            cv2.line(frame, points[i], points[i + 1], color, thickness)

        for i, point in enumerate(points):
            if i == 0:
                cv2.circle(frame, point, 8, (100, 100, 150), -1)
                cv2.circle(frame, point, 8, (150, 150, 255), 2)
                continue

            beat_label = beat_labels[i]
            is_current = i == current_beat
            is_next = str(current_beat + 1) == beat_label.split()[0] if beat_label else False

            if is_current:
                circle_color = (0, 200, 0)
                radius = 14
            elif is_next:
                circle_color = (0, 150, 0)
                radius = 12
            else:
                circle_color = (50, 50, 150)
                radius = 10

            cv2.circle(frame, point, radius, circle_color, -1)
            cv2.circle(frame, point, radius, (100, 100, 255), 2)
            cv2.putText(frame, beat_label, (point[0] - 45, point[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)

        return frame

    def draw_musical_notation(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        x = self.const.WINDOW_WIDTH - self.const.NOTATION_WIDTH - 20
        y = 80

        cv2.rectangle(frame, (x, y), (x + self.const.NOTATION_WIDTH, y + self.const.NOTATION_HEIGHT),
                      (245, 245, 220), -1)
        cv2.rectangle(frame, (x, y), (x + self.const.NOTATION_WIDTH, y + self.const.NOTATION_HEIGHT),
                      (100, 100, 100), 2)

        notation = self.notation_display.create_notation_text(
            stats['score_name'],
            stats['time_signature'],
            stats['current_measure'],
            stats['current_beat'],
            stats['beats_per_measure'],
            stats['notation_events'],
            stats['show_completion_banner']
        )

        frame[y:y + notation.shape[0], x:x + notation.shape[1]] = notation

        return frame

    def draw_score_info_panel(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        x = self.const.WINDOW_WIDTH - self.const.NOTATION_WIDTH - 20
        y = 80 + self.const.NOTATION_HEIGHT + 10

        info_panel = self.notation_display.create_score_info_display(
            stats['score_name'],
            stats['composer'],
            stats['time_signature'],
            stats['description'],
            stats['show_completion_banner']
        )

        if y + info_panel.shape[0] < self.const.WINDOW_HEIGHT - 20:
            frame[y:y + info_panel.shape[0], x:x + info_panel.shape[1]] = info_panel

        return frame

    def draw_info_panel(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 100), (15 + self.const.INFO_PANEL_WIDTH,
                                           100 + self.const.INFO_PANEL_HEIGHT),
                      (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        y_offset = 125

        if stats['show_completion_banner']:
            cv2.putText(frame, "CONGRATULATIONS!", (20, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "You've completed the piece!", (20, y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Press [1]/[2] for next score", (20, y_offset + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            cv2.putText(frame, "or [r] to replay", (20, y_offset + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            return frame

        cv2.putText(frame, "STATUS", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame, f"Score: {stats['score_id']} - {stats['score_name']}",
                    (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(frame, f"Time: {stats['time_signature']} ({stats['beats_per_measure']} beats)",
                    (20, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"Tempo: {stats['bpm']} / {stats['suggested_tempo']} BPM",
                    (20, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Beats: {stats['beats']}  |  Measures: {stats['measures']}",
                    (20, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Current Beat: {stats['current_beat']} / {stats['beats_per_measure']}",
                    (20, y_offset + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(frame, f"Progress: {stats['score_pos']}/{stats['score_total']}",
                    (20, y_offset + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Show FPS only when actively conducting
        if stats['beats'] > 0 and not stats['show_completion_banner']:
            cv2.putText(frame, f"FPS: {stats.get('fps', 0)}", (20, y_offset + 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def draw_keyboard_indicator(self, frame: np.ndarray, next_beat: str,
                                current_beat: int, beats_per_measure: int,
                                show_completion: bool = False) -> np.ndarray:
        x = self.const.WINDOW_WIDTH - self.const.SCORE_PANEL_WIDTH - self.const.SCORE_PANEL_X_OFFSET
        y = self.const.WINDOW_HEIGHT - self.const.KEYBOARD_PANEL_HEIGHT - self.const.SCORE_PANEL_X_OFFSET - 100

        cv2.rectangle(frame, (x, y), (self.const.WINDOW_WIDTH - 40,
                                      self.const.WINDOW_HEIGHT - 20),
                      (30, 30, 50), -1)
        cv2.rectangle(frame, (x, y), (self.const.WINDOW_WIDTH - 40,
                                      self.const.WINDOW_HEIGHT - 20),
                      (100, 100, 150), 2)

        if show_completion:
            cv2.putText(frame, "SCORE COMPLETE!", (x + 60, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Press [1]/[2] or [r]", (x + 70, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            return frame

        next_beat_num = current_beat + 1 if current_beat < beats_per_measure else 1
        cv2.putText(frame, f"NEXT BEAT: {next_beat_num} / {beats_per_measure}",
                    (x + 60, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(frame, "PRESS:", (x + 30, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        key_map = {
            "DOWN": (110, 40, "DOWN"),
            "LEFT": (70, 65, "LEFT"),
            "RIGHT": (150, 65, "RIGHT"),
            "UP": (110, 40, "UP")
        }

        if next_beat in key_map:
            x_off, y_off, key_name = key_map[next_beat]
            kx = x + x_off
            ky = y + y_off
            cv2.rectangle(frame, (kx, ky), (kx + 80, ky + 28), (0, 100, 0), -1)
            cv2.rectangle(frame, (kx, ky), (kx + 80, ky + 28), (0, 255, 0), 2)
            cv2.putText(frame, key_name, (kx + 20, ky + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return frame

    def draw_error(self, frame: np.ndarray, error_msg: str, error_time: float) -> np.ndarray:
        if error_msg and time.time() - error_time < 1.2:
            center_x = self.const.WINDOW_WIDTH // 2
            center_y = self.const.WINDOW_HEIGHT // 2 + 200
            (text_w, text_h), _ = cv2.getTextSize(error_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (center_x - text_w // 2 - 10, center_y - text_h - 5),
                          (center_x + text_w // 2 + 10, center_y + 10),
                          (0, 0, 100), -1)
            cv2.rectangle(frame, (center_x - text_w // 2 - 10, center_y - text_h - 5),
                          (center_x + text_w // 2 + 10, center_y + 10),
                          (0, 0, 255), 2)
            cv2.putText(frame, error_msg, (center_x - text_w // 2, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def draw_beat_flash(self, frame: np.ndarray, is_flash: bool, beat_name: str,
                        beat_number: int, beats_per_measure: int) -> np.ndarray:
        if is_flash:
            center_x = 350
            center_y = self.const.WINDOW_HEIGHT // 2 - 50
            overlay = frame.copy()
            cv2.circle(overlay, (center_x, center_y + 80), 60, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            text = f"BEAT {beat_number}/{beats_per_measure}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, text, (center_x - text_w // 2, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def draw_controls(self, frame: np.ndarray, fps: int, show_completion: bool = False) -> np.ndarray:
        controls_text = "[1]Prev  [2]Next  [m]Metronome  [r]Reset  [s]Save  [q]Quit"
        (text_w, _), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, controls_text,
                    (self.const.WINDOW_WIDTH // 2 - text_w // 2, self.const.WINDOW_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        return frame


# ============================================
# MAIN APPLICATION
# ============================================

class ConductorApp:
    def __init__(self):
        self.conductor = Conductor()
        self.visualizer = Visualizer()
        self._error_msg = ""
        self._error_time = 0
        self._fps = 0
        self._frame_count = 0
        self._fps_time = time.time()

        self._setup_window()
        self._print_welcome()

    def _setup_window(self) -> None:
        cv2.namedWindow('Virtual Choir Conductor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Virtual Choir Conductor',
                         VisualConstants.WINDOW_WIDTH,
                         VisualConstants.WINDOW_HEIGHT)

    def _print_welcome(self) -> None:
        print("=" * 70)
        print("VIRTUAL CHOIR CONDUCTOR")
        print("Completion Banner Appears After Last Note Finishes")
        print("=" * 70)
        print("\nAvailable Scores:")
        for score in MusicScoreLibrary.get_all():
            notes = len(score.musical_events)
            print(f"  {score.id}. {score.name} ({score.composer}) - {score.time_signature} ({notes} beats)")
        print("\n[CONTROLS]")
        print("  Arrow Keys - Conduct (follow the pattern shown)")
        print("  [1] / [2] - Previous/Next score")
        print("  [m] - Metronome on/off")
        print("  [r] - Reset current score")
        print("  [s] - Save results")
        print("  [q] - Quit")
        print("=" * 70)

        stats = self.conductor.stats
        print(f"\nSTARTING WITH:")
        print(f"  Score {stats['score_id']}: {stats['score_name']} by {stats['composer']}")
        print(f"  Time Signature: {stats['time_signature']}")
        print(f"  Suggested Tempo: {stats['suggested_tempo']} BPM")
        print("=" * 70)

    def _process_input(self, key: int) -> bool:
        direction = BeatDirection.from_keycode(key)
        if direction:
            success, result = self.conductor.register_beat(direction)
            if success:
                print(f"  {result}")
            else:
                print(f"  X {result}")
                self._error_msg = result
                self._error_time = time.time()
            return True

        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.conductor.reset()
            self._error_msg = ""
        elif key == ord('1'):
            self.conductor.previous_score()
        elif key == ord('2'):
            self.conductor.next_score()
        elif key == ord('m'):
            self.conductor.toggle_metronome()
        elif key == ord('s'):
            self._save_results()

        return True

    def _save_results(self) -> None:
        stats = self.conductor.stats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conductor_results_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write("VIRTUAL CHOIR CONDUCTOR - SESSION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Score: {stats['score_id']} - {stats['score_name']}\n")
            f.write(f"Composer: {stats['composer']}\n")
            f.write(f"Time Signature: {stats['time_signature']}\n")
            f.write(f"  {stats['time_signature_note']}\n")
            f.write(f"Pattern: {stats['pattern_display']}\n")
            f.write(f"Suggested Tempo: {stats['suggested_tempo']} BPM\n")
            f.write(f"Average BPM: {stats['bpm']}\n")
            f.write(f"Total Beats: {stats['beats']}\n")
            f.write(f"Total Measures: {stats['measures']}\n")
            f.write(f"Progress: {stats['score_pos']}/{stats['score_total']}\n")
            f.write("=" * 50 + "\n")

        print(f"\n[Saved session results to {filename}]")

    def _update_fps(self) -> int:
        self._frame_count += 1
        if time.time() - self._fps_time > 1.0:
            self._fps = self._frame_count
            self._frame_count = 0
            self._fps_time = time.time()
        return self._fps

    def _create_frame(self) -> np.ndarray:
        stats = self.conductor.stats
        frame = self.visualizer.create_background()
        self.visualizer.draw_title(frame)

        # Add FPS to stats only if conducting
        stats_with_fps = stats.copy()
        if stats['beats'] > 0 and not stats['show_completion_banner']:
            stats_with_fps['fps'] = self._update_fps()
        else:
            stats_with_fps['fps'] = 0

        frame = self.visualizer.draw_pattern_guide(
            frame, stats['time_signature'],
            self.conductor.expected_pattern,
            stats['next_beat'],
            stats['current_beat'],
            stats['show_completion_banner']
        )
        frame = self.visualizer.draw_musical_notation(frame, stats_with_fps)
        frame = self.visualizer.draw_score_info_panel(frame, stats_with_fps)
        frame = self.visualizer.draw_info_panel(frame, stats_with_fps)
        frame = self.visualizer.draw_keyboard_indicator(
            frame, stats['next_beat'],
            stats['current_beat'],
            stats['beats_per_measure'],
            stats['show_completion_banner']
        )
        frame = self.visualizer.draw_error(frame, self._error_msg, self._error_time)
        frame = self.visualizer.draw_beat_flash(
            frame, self.conductor.is_beat_flash, self.conductor.last_beat_name,
            stats['current_beat'], stats['beats_per_measure']
        )
        frame = self.visualizer.draw_controls(frame, stats_with_fps['fps'])
        return frame

    def _print_summary(self) -> None:
        final_stats = self.conductor.get_final_stats()
        print("\n" + "=" * 55)
        print("SESSION SUMMARY")
        print("=" * 55)
        print(f"  Score: {final_stats['score_id']} - {final_stats['score_name']}")
        print(f"  Composer: {final_stats['composer']}")
        print(f"  Time Signature: {final_stats['time_signature']}")
        print(f"  Suggested Tempo: {final_stats['suggested_tempo']} BPM")
        print(f"  Your Average BPM: {final_stats['final_bpm']}")
        print(f"  Total Beats Conducted: {final_stats['total_beats']}")
        print(f"  Total Measures: {final_stats['total_measures']}")
        print("=" * 55)
        print("Thank you for conducting!")

    def run(self) -> None:
        while True:
            frame = self._create_frame()
            cv2.imshow('Virtual Choir Conductor', frame)
            key = cv2.waitKey(50) & 0xFF
            if not self._process_input(key):
                break

        cv2.destroyAllWindows()
        self._print_summary()


def main() -> None:
    app = ConductorApp()
    app.run()


if __name__ == "__main__":
    main()