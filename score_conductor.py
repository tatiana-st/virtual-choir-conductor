#!/usr/bin/env python3
"""
SATB Choir Conductor Simulator

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

class Voice(Enum):
    SOPRANO = "Soprano"
    ALTO = "Alto"
    TENOR = "Tenor"
    BASS = "Bass"


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


@dataclass
class VisualConstants:
    WINDOW_WIDTH: int = 1600
    WINDOW_HEIGHT: int = 1100
    PATTERN_SIZE: int = 70
    SCORE_PANEL_WIDTH: int = 350
    SCORE_PANEL_HEIGHT: int = 450
    SCORE_PANEL_X_OFFSET: int = 20
    SCORE_PANEL_Y_OFFSET: int = 80
    KEYBOARD_PANEL_HEIGHT: int = 120
    INFO_PANEL_WIDTH: int = 340
    INFO_PANEL_HEIGHT: int = 340
    NOTATION_WIDTH: int = 720
    NOTATION_HEIGHT: int = 620


# ============================================
# SATB CHORD AND VOICE DATA
# ============================================

@dataclass
class SATBChord:
    soprano: str
    alto: str
    tenor: str
    bass: str
    beat_position: int
    duration: float
    chord_name: str = ""

    def get_all_notes(self) -> List[Tuple[str, str]]:
        return [
            (Voice.SOPRANO.value, self.soprano),
            (Voice.ALTO.value, self.alto),
            (Voice.TENOR.value, self.tenor),
            (Voice.BASS.value, self.bass),
        ]


# ============================================
# AUDIO GENERATOR
# ============================================

class SATBAudioGenerator:
    FREQUENCIES: Dict[str, float] = {
        'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61,
        'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
        'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23,
        'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
        'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46,
        'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'B5': 987.77, 'C6': 1046.50,
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._current_audio = None
        self.voice_volumes = {
            Voice.SOPRANO.value: 0.35,
            Voice.ALTO.value: 0.30,
            Voice.TENOR.value: 0.30,
            Voice.BASS.value: 0.35
        }
        self.mute_voices = set()

    def set_voice_volume(self, voice: str, volume: float):
        self.voice_volumes[voice] = np.clip(volume, 0, 0.5)

    def toggle_mute_voice(self, voice: str):
        if voice in self.mute_voices:
            self.mute_voices.remove(voice)
        else:
            self.mute_voices.add(voice)

    def generate_tone(self, frequency: float, duration: float, volume: float = 0.3) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)
        fade_samples = int(0.01 * self.sample_rate)
        if len(wave) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out
        wave = wave * volume
        wave = np.clip(wave, -0.99, 0.99)
        return (wave * 32767).astype(np.int16)

    def play_chord(self, chord: SATBChord, duration: float) -> None:
        if len(chord.get_all_notes()) == 0:
            return

        t = np.linspace(0, duration, int(self.sample_rate * duration))
        combined = np.zeros(len(t))

        for voice_name, note in chord.get_all_notes():
            if voice_name in self.mute_voices:
                continue
            if note in self.FREQUENCIES:
                freq = self.FREQUENCIES[note]
                volume = self.voice_volumes.get(voice_name, 0.3)
                wave = volume * np.sin(2 * np.pi * freq * t)
                combined += wave

        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.8

        fade_samples = int(0.008 * self.sample_rate)
        if len(combined) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            combined[:fade_samples] *= fade_in
            combined[-fade_samples:] *= fade_out

        audio = (combined * 32767).astype(np.int16)

        if self._current_audio:
            self._current_audio.stop()
        self._current_audio = sa.play_buffer(audio, 1, 2, self.sample_rate)

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
# SATB CHORAL SCORES
# ============================================

@dataclass
class SATBScore:
    id: int
    name: str
    composer: str
    time_signature: str
    beats_per_measure: int
    pattern: List[str]
    pattern_display: str
    description: str
    chords: List[SATBChord]
    tempo_bpm: int
    lyrics: List[str]


class ChoralLibrary:

    @staticmethod
    def create_ave_maria() -> SATBScore:
        chords = [
            SATBChord('A5', 'E5', 'C4', 'A3', 1, 1.5, "Ave"),
            SATBChord('G5', 'D5', 'B3', 'G3', 2, 1.0, "Ma-"),
            SATBChord('E5', 'C5', 'G3', 'C3', 3, 1.5, "ri-a"),
            SATBChord('D5', 'A4', 'F3', 'D3', 4, 1.0, "gra-"),
            SATBChord('C5', 'G4', 'E3', 'C3', 1, 2.0, "ti-a"),
            SATBChord('E5', 'C5', 'G3', 'C3', 2, 1.0, "ple-"),
            SATBChord('D5', 'B4', 'G3', 'G3', 3, 1.0, "na"),
            SATBChord('C5', 'A4', 'F3', 'F3', 4, 1.0, "do-"),
            SATBChord('G4', 'E4', 'C4', 'C3', 1, 2.0, "mi-nus"),
            SATBChord('C5', 'G4', 'E4', 'C3', 2, 1.0, "A-"),
            SATBChord('G4', 'E4', 'C4', 'C3', 3, 1.0, "men"),
            SATBChord('C5', 'C5', 'G4', 'C3', 4, 2.0, ""),
        ]
        lyrics = ["Ave", "Ma-", "ri-a", "gra-", "ti-a", "ple-", "na", "do-", "mi-nus", "A-", "men", ""]
        return SATBScore(1, "Ave Maria", "Traditional", "4/4", 4,
                         ["DOWN", "LEFT", "RIGHT", "UP"], "DOWN -> LEFT -> RIGHT -> UP",
                         "Beautiful SATB setting of Ave Maria", chords, 80, lyrics)

    @staticmethod
    def create_amazing_grace() -> SATBScore:
        chords = [
            SATBChord('E5', 'C5', 'G4', 'C3', 1, 1.5, "A-"), SATBChord('D5', 'B4', 'G4', 'G3', 2, 0.5, "ma-"),
            SATBChord('C5', 'A4', 'E4', 'C3', 3, 1.0, "zing"), SATBChord('G4', 'E4', 'C4', 'C3', 4, 0.5, "grace"),
            SATBChord('E5', 'C5', 'G4', 'C3', 1, 1.0, "how"), SATBChord('D5', 'B4', 'G4', 'G3', 2, 0.5, "sweet"),
            SATBChord('C5', 'A4', 'F4', 'F3', 3, 1.0, "the"), SATBChord('G4', 'E4', 'C4', 'C3', 4, 2.0, "sound"),
            SATBChord('C5', 'G4', 'E4', 'C3', 1, 1.0, "that"), SATBChord('D5', 'A4', 'F4', 'D3', 2, 0.5, "saved"),
            SATBChord('E5', 'C5', 'G4', 'E3', 3, 1.0, "a"), SATBChord('C5', 'G4', 'E4', 'C3', 4, 0.5, "wretch"),
            SATBChord('G4', 'E4', 'C4', 'C3', 1, 1.0, "like"), SATBChord('A4', 'F4', 'D4', 'D3', 2, 0.5, "me"),
            SATBChord('G4', 'E4', 'C4', 'C3', 3, 1.0, "I"), SATBChord('C5', 'G4', 'E4', 'C3', 4, 2.0, "once"),
        ]
        lyrics = ["A-", "ma-", "zing", "grace", "how", "sweet", "the", "sound",
                  "that", "saved", "a", "wretch", "like", "me", "I", "once"]
        return SATBScore(2, "Amazing Grace", "John Newton", "3/4", 3,
                         ["DOWN", "RIGHT", "UP"], "DOWN -> RIGHT -> UP",
                         "Traditional hymn in SATB arrangement", chords, 90, lyrics)

    @staticmethod
    def create_alleluia() -> SATBScore:
        chords = [
            SATBChord('G5', 'E5', 'C5', 'C3', 1, 0.8, "Al-"), SATBChord('A5', 'F5', 'D5', 'D3', 2, 0.8, "le-"),
            SATBChord('G5', 'E5', 'C5', 'C3', 3, 0.8, "lu-"), SATBChord('C6', 'G5', 'E5', 'C3', 4, 1.2, "ia"),
            SATBChord('G5', 'E5', 'C5', 'C3', 1, 0.8, "Al-"), SATBChord('A5', 'F5', 'D5', 'D3', 2, 0.8, "le-"),
            SATBChord('B5', 'G5', 'E5', 'E3', 3, 0.8, "lu-"), SATBChord('C6', 'G5', 'E5', 'C3', 4, 1.2, "ia"),
            SATBChord('G5', 'E5', 'C5', 'C3', 1, 1.5, ""), SATBChord('C6', 'C6', 'G5', 'C3', 2, 2.0, "A-men"),
        ]
        lyrics = ["Al-", "le-", "lu-", "ia", "Al-", "le-", "lu-", "ia", "", "A-men"]
        return SATBScore(3, "Alleluia", "Traditional", "4/4", 4,
                         ["DOWN", "LEFT", "RIGHT", "UP"], "DOWN -> LEFT -> RIGHT -> UP",
                         "Festive SATB alleluia", chords, 100, lyrics)

    @staticmethod
    def get_all() -> List[SATBScore]:
        return [ChoralLibrary.create_ave_maria(),
                ChoralLibrary.create_amazing_grace(),
                ChoralLibrary.create_alleluia()]


# ============================================
# SATB SCORE PLAYER
# ============================================

class SATBScorePlayer:
    def __init__(self, initial_id: int = 1):
        self._scores = ChoralLibrary.get_all()
        self._current_index = self._find_index_by_id(initial_id)
        self._chord_index: int = 0
        self._completed: bool = False
        self._load_current_score()

    def _find_index_by_id(self, score_id: int) -> int:
        for i, score in enumerate(self._scores):
            if score.id == score_id:
                return i
        return 0

    def _load_current_score(self) -> None:
        self.current_score = self._scores[self._current_index]
        self._chord_index = 0
        self._completed = False

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
            "position": self._chord_index,
            "total": len(self.current_score.chords),
            "completed": self._completed,
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

    def get_upcoming_chords(self, num: int = 3) -> List[Tuple[SATBChord, str]]:
        chords = []
        for i in range(self._chord_index, min(self._chord_index + num, len(self.current_score.chords))):
            chords.append((self.current_score.chords[i],
                           self.current_score.lyrics[i] if i < len(self.current_score.lyrics) else ""))
        return chords

    def get_current_lyric(self) -> str:
        if self._chord_index >= len(self.current_score.lyrics):
            return ""
        return self.current_score.lyrics[self._chord_index]

    def advance_to_next_chord(self) -> Tuple[Optional[SATBChord], str, bool]:
        if self._completed:
            return None, "", False

        chord = self.current_score.chords[self._chord_index]
        lyric = self.current_score.lyrics[self._chord_index] if self._chord_index < len(
            self.current_score.lyrics) else ""
        is_last = (self._chord_index == len(self.current_score.chords) - 1)
        self._chord_index += 1
        if self._chord_index >= len(self.current_score.chords):
            self._completed = True
        return chord, lyric, is_last

    def next_score(self) -> Dict:
        self._current_index = (self._current_index + 1) % len(self._scores)
        self._load_current_score()
        return self.info

    def previous_score(self) -> Dict:
        self._current_index = (self._current_index - 1) % len(self._scores)
        self._load_current_score()
        return self.info

    def reset(self) -> None:
        self._chord_index = 0
        self._completed = False


# ============================================
# VOICE STAFF RENDERER
# ============================================

class VoiceStaffRenderer:
    TREBLE_POSITIONS = {
        'C4': 4, 'D4': 3, 'E4': 2, 'F4': 1, 'G4': 0,
        'A4': -1, 'B4': -2, 'C5': -3, 'D5': -4, 'E5': -5,
        'F5': -6, 'G5': -7, 'A5': -8, 'B5': -9, 'C6': -10,
    }

    BASS_POSITIONS = {
        'C3': 3, 'D3': 2, 'E3': 1, 'F3': 0, 'G3': -1, 'A3': -2, 'B3': -3,
        'C4': -4, 'D4': -5, 'E4': -6, 'F4': -7, 'G4': -8,
    }

    @staticmethod
    def draw_note(frame: np.ndarray, note: str, x: int, staff_top: int,
                  line_spacing: float, is_treble_clef: bool, is_current: bool = False) -> np.ndarray:
        if not note or len(note) < 2:
            return frame

        positions = VoiceStaffRenderer.TREBLE_POSITIONS if is_treble_clef else VoiceStaffRenderer.BASS_POSITIONS

        if note in positions:
            offset = positions[note]
            y_pos = int(staff_top + (4 - offset) * line_spacing)

            color = (0, 200, 0) if is_current else (0, 0, 0)
            thickness = -1 if is_current else 1

            cv2.circle(frame, (x, y_pos), 7, color, thickness)
            cv2.line(frame, (x + 6, y_pos), (x + 6, y_pos - 20), (0, 0, 0), 1)

            if offset < 0:
                ledger_line_y = int(staff_top - line_spacing)
                for _ in range(0, -offset, 2):
                    cv2.line(frame, (x - 8, ledger_line_y), (x + 15, ledger_line_y), (0, 0, 0), 1)
                    ledger_line_y -= int(line_spacing)
            elif offset > 4:
                ledger_line_y = int(staff_top + 4 * line_spacing + line_spacing)
                for _ in range(offset - 4, 0, -2):
                    cv2.line(frame, (x - 8, ledger_line_y), (x + 15, ledger_line_y), (0, 0, 0), 1)
                    ledger_line_y += int(line_spacing)

        return frame


# ============================================
# SATB NOTATION DISPLAY
# ============================================

class SATBNotationDisplay:
    def __init__(self):
        self.staff_renderer = VoiceStaffRenderer()

    def draw_satb_score(self, frame: np.ndarray, score: SATBScore, current_chord_index: int,
                        chords: List[Tuple[SATBChord, str]], x: int, y: int, w: int, h: int) -> np.ndarray:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 220), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)

        cv2.putText(frame, f"{score.name} - {score.composer}", (x + 20, y + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Time: {score.time_signature}  |  Tempo: {score.tempo_bpm} BPM  |  [ and ] to change piece",
                    (x + 20, y + 50), cv2.FONT_HERSHEY_DUPLEX, 0.4, (80, 80, 80), 1)

        voice_y_start = y + 85
        voice_height = 125
        staff_height = 75
        separator_thickness = 3

        voice_colors = {
            Voice.SOPRANO.value: (255, 235, 235),
            Voice.ALTO.value: (235, 255, 235),
            Voice.TENOR.value: (235, 235, 255),
            Voice.BASS.value: (255, 255, 235),
        }

        voice_spacing = voice_height + 15
        voice_defs = [
            (Voice.SOPRANO.value, voice_y_start, True),
            (Voice.ALTO.value, voice_y_start + voice_spacing, True),
            (Voice.TENOR.value, voice_y_start + voice_spacing * 2, False),
            (Voice.BASS.value, voice_y_start + voice_spacing * 3, False),
        ]

        for idx in range(1, 4):
            sep_y = voice_y_start + voice_spacing * idx - 8
            cv2.line(frame, (x + 10, sep_y), (x + w - 10, sep_y), (100, 100, 150), separator_thickness)

        for voice_name, vy, is_treble in voice_defs:
            bg_color = voice_colors.get(voice_name, (240, 240, 240))
            cv2.rectangle(frame, (x + 10, vy), (x + w - 10, vy + voice_height - 5), bg_color, -1)
            cv2.rectangle(frame, (x + 10, vy), (x + w - 10, vy + voice_height - 5), (120, 120, 150), 2)

            cv2.putText(frame, voice_name, (x + 18, vy + 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 100), 1)

            staff_y = vy + 42
            line_spacing = staff_height / 4

            for i in range(5):
                line_y = int(staff_y + i * line_spacing)
                cv2.line(frame, (x + 95, line_y), (x + w - 20, line_y), (0, 0, 0), 1)

            clef_x = x + 72
            if is_treble:
                cv2.putText(frame, "G", (clef_x, int(staff_y + 2 * line_spacing + 8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
            else:
                cv2.putText(frame, "F", (clef_x + 2, int(staff_y + 2 * line_spacing + 8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

            if voice_name == Voice.SOPRANO.value:
                cv2.putText(frame, score.time_signature, (x + 90, int(staff_y + 2 * line_spacing + 8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            note_pos_x = x + 130
            note_spacing = 42

            for chord_idx, (chord, lyric) in enumerate(chords[:6]):
                if voice_name == Voice.SOPRANO.value:
                    note = chord.soprano
                elif voice_name == Voice.ALTO.value:
                    note = chord.alto
                elif voice_name == Voice.TENOR.value:
                    note = chord.tenor
                else:
                    note = chord.bass

                is_current = (chord_idx == 0)

                frame = self.staff_renderer.draw_note(
                    frame, note, note_pos_x, staff_y, line_spacing, is_treble, is_current
                )

                if lyric and is_current and len(lyric) > 0:
                    cv2.putText(frame, lyric, (note_pos_x - 12, int(staff_y + staff_height + 18)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 0, 0), 1)

                cv2.putText(frame, str(chord.beat_position), (note_pos_x - 3, int(staff_y + staff_height + 35)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, (80, 80, 80), 1)

                note_pos_x += note_spacing

            if len(chords) > 1:
                prev_x = x + 130
                for chord_idx in range(1, min(len(chords), 6)):
                    current_x = x + 130 + chord_idx * note_spacing
                    cv2.line(frame, (prev_x + 15, int(staff_y + 2 * line_spacing)),
                             (current_x, int(staff_y + 2 * line_spacing)), (100, 100, 100), 1)
                    prev_x = current_x

        cv2.line(frame, (x + 90, voice_y_start), (x + 90, voice_y_start + voice_spacing * 4 - 10), (0, 0, 0), 4)

        return frame

    @staticmethod
    def draw_voice_controls(frame: np.ndarray, audio_gen: SATBAudioGenerator,
                            x: int, y: int, w: int, h: int) -> np.ndarray:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 55), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 170), 2)

        cv2.putText(frame, "VOICE CONTROLS", (x + w // 2 - 70, y + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 0), 1)

        voices = [Voice.SOPRANO.value, Voice.ALTO.value, Voice.TENOR.value, Voice.BASS.value]
        button_y = y + 48
        button_height = 30

        for i, voice in enumerate(voices):
            btn_y = button_y + i * (button_height + 8)

            cv2.rectangle(frame, (x + 15, btn_y), (x + 155, btn_y + button_height), (55, 55, 75), -1)
            volume = audio_gen.voice_volumes.get(voice, 0.3)
            vol_width = int(volume * 140)
            cv2.rectangle(frame, (x + 15, btn_y), (x + 15 + vol_width, btn_y + button_height), (0, 150, 0), -1)

            cv2.putText(frame, voice, (x + 170, btn_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 200), 1)

            is_muted = voice in audio_gen.mute_voices
            mute_color = (0, 0, 180) if is_muted else (0, 180, 0)
            status_text = "MUTED" if is_muted else "ON"
            cv2.putText(frame, status_text, (x + 250, btn_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, mute_color, 1)

        cv2.putText(frame, "1-4: Mute  |  Shift+1-4: Volume +  |  [: Prev  |  ]: Next  |  a: RESYNC",
                    (x + 15, button_y + 4 * (button_height + 8) + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 200), 1)

        return frame


# ============================================
# SATB CONDUCTOR
# ============================================

class SATBConductor:
    MIN_BEAT_INTERVAL: float = 0.2
    MAX_BEAT_INTERVAL: float = 2.0
    MIN_BPM_INTERVAL: float = 0.25
    MAX_BPM_INTERVAL: float = 1.5
    BEAT_HISTORY_SIZE: int = 10
    TEMPO_TOLERANCE: float = 0.3

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
        self._tempo_error: bool = False
        self._tempo_error_msg: str = ""
        self._tempo_error_time: float = 0
        self._resync_message_time: float = 0
        self._show_resync_success: bool = False

        self._score_player = SATBScorePlayer()
        self._audio = SATBAudioGenerator()
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
                self._tempo_error = True
                self._tempo_error_msg = f"TOO FAST! Target: {self._score_player.suggested_tempo} BPM"
                self._tempo_error_time = now
                return False, self._tempo_error_msg
            if interval > self.MAX_BEAT_INTERVAL and self._beat_count > 0:
                self._tempo_error = True
                self._tempo_error_msg = f"TOO SLOW! Target: {self._score_player.suggested_tempo} BPM"
                self._tempo_error_time = now
                return False, self._tempo_error_msg

            if self._score_player.suggested_tempo > 0:
                bpm_estimate = int(60.0 / interval)
                deviation = abs(bpm_estimate - self._score_player.suggested_tempo) / self._score_player.suggested_tempo
                if deviation > self.TEMPO_TOLERANCE and self._beat_count > 4:
                    self._tempo_error = True
                    self._tempo_error_msg = f"TEMPO OFF! Target: {self._score_player.suggested_tempo} BPM, You: {bpm_estimate} BPM"
                    self._tempo_error_time = now
                    return False, self._tempo_error_msg

        self._tempo_error = False
        return True, ""

    def resync_tempo(self) -> None:
        self._beat_times.clear()
        self._beat_phase = 0
        self._expected_next = self._expected_pattern[0] if self._expected_pattern else "DOWN"
        self._tempo_error = False
        self._current_bpm = 0
        self._last_beat_time = 0
        self._show_resync_success = True
        self._resync_message_time = time.time()

        print("\n" + "=" * 50)
        print("RESYNC COMPLETE!")
        print(f"  Beat phase reset to: {self._expected_next}")
        print(f"  Music continues from current position")
        print("  Start conducting from the next DOWNBEAT")
        print("=" * 50)

    @property
    def show_resync_message(self) -> bool:
        if not self._show_resync_success:
            return False
        if time.time() - self._resync_message_time > 2.0:
            self._show_resync_success = False
            return False
        return True

    def register_beat(self, direction: BeatDirection) -> Tuple[bool, str]:
        if self._completion_banner_visible:
            return False, "SCORE COMPLETE! Press [ and ] for new score or [r] to replay"

        now = time.time()
        beat_name = direction.value

        tempo_ok, tempo_msg = self._check_tempo(now)
        if not tempo_ok:
            return False, tempo_msg + " | Press 'a' to RESYNC"

        if beat_name != self._expected_next:
            return False, f"WRONG PATTERN (expected {self._expected_next}) | Press 'a' to RESYNC"

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

        chord, lyric, is_last = self._score_player.advance_to_next_chord()
        if chord:
            self._audio.play_chord(chord, chord.duration)
            if self.metronome_enabled:
                is_downbeat = (chord.beat_position == 1)
                self._audio.play_metronome_click(is_downbeat)

            msg = f"[OK] Beat {chord.beat_position} - {lyric}"

            if is_last:
                def show_banner():
                    time.sleep(chord.duration + 0.1)
                    self._show_completion_banner()

                threading.Thread(target=show_banner, daemon=True).start()
                return True, msg + " (Last chord - completing soon!)"

            return True, msg

        return True, "SCORE COMPLETE"

    def _show_completion_banner(self) -> None:
        self._completion_banner_visible = True
        self._completion_banner_start_time = time.time()
        self._audio.play_completion_fanfare()

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
        if not self._completion_banner_visible:
            return False
        if time.time() - self._completion_banner_start_time > 4.0:
            self._completion_banner_visible = False
            return False
        return True

    @property
    def has_tempo_error(self) -> bool:
        return self._tempo_error and (time.time() - self._tempo_error_time) < 2.0

    @property
    def tempo_error_msg(self) -> str:
        return self._tempo_error_msg

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
            "next_beat": self._expected_next,
            "current_beat": self.current_beat_number,
            "score_pos": info["position"],
            "score_total": info["total"],
            "is_completed": self.is_completed,
            "show_completion_banner": self.show_completion_banner,
            "has_tempo_error": self.has_tempo_error,
            "tempo_error_msg": self.tempo_error_msg,
            "show_resync_message": self.show_resync_message,
            "current_lyric": self._score_player.get_current_lyric(),
            "upcoming_chords": self._score_player.get_upcoming_chords(4),
        }

    @property
    def expected_pattern(self) -> List[str]:
        return self._expected_pattern

    def next_score(self) -> None:
        self._score_player.next_score()
        self._update_pattern()
        self._beat_count = 0
        self._measure_count = 0
        self._beat_times.clear()
        self._current_bpm = 0
        self._completion_banner_visible = False
        self._tempo_error = False
        self._print_score_info()

    def previous_score(self) -> None:
        self._score_player.previous_score()
        self._update_pattern()
        self._beat_count = 0
        self._measure_count = 0
        self._beat_times.clear()
        self._current_bpm = 0
        self._completion_banner_visible = False
        self._tempo_error = False
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
        self._tempo_error = False
        print("\n[RESET] Score restarted from beginning!")

    def _print_score_info(self) -> None:
        info = self._score_player.info
        print(f"\n{'=' * 60}")
        print(f"SATB SCORE {info['id']}: {info['name']} (by {info['composer']})")
        print(f"{'=' * 60}")
        print(f"  Time Signature: {info['time_signature']}")
        print(f"  Suggested Tempo: {info['tempo_bpm']} BPM")
        print(f"  Conducting Pattern: {info['pattern_display']}")
        print(f"  {info['description']}")
        print(f"  Total Chords: {info['total']}")
        print(f"{'=' * 60}")
        print("TIP: Press '[' for PREVIOUS piece | Press ']' for NEXT piece")
        print("      Press 'a' to RESYNC (fix tempo without restarting)")
        print("      Press 'r' to RESET (restart the score from beginning)")

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

    def mute_voice(self, voice_idx: int):
        voices = [Voice.SOPRANO.value, Voice.ALTO.value, Voice.TENOR.value, Voice.BASS.value]
        if 0 <= voice_idx < len(voices):
            self._audio.toggle_mute_voice(voices[voice_idx])
            status = "MUTED" if voices[voice_idx] in self._audio.mute_voices else "UNMUTED"
            print(f"\n[VOICE {voices[voice_idx]}: {status}]")

    def adjust_voice_volume(self, voice_idx: int, delta: float):
        voices = [Voice.SOPRANO.value, Voice.ALTO.value, Voice.TENOR.value, Voice.BASS.value]
        if 0 <= voice_idx < len(voices):
            current = self._audio.voice_volumes.get(voices[voice_idx], 0.3)
            new_vol = current + delta
            self._audio.set_voice_volume(voices[voice_idx], new_vol)
            print(f"\n[VOICE {voices[voice_idx]} VOLUME: {new_vol:.2f}]")


# ============================================
# SATB VISUALIZER
# ============================================

class SATBVisualizer:
    def __init__(self, constants: VisualConstants = VisualConstants()):
        self.const = constants
        self.notation_display = SATBNotationDisplay()

    def create_background(self) -> np.ndarray:
        frame = np.zeros((self.const.WINDOW_HEIGHT, self.const.WINDOW_WIDTH, 3), dtype=np.uint8)
        for x in range(0, self.const.WINDOW_WIDTH, 50):
            cv2.line(frame, (x, 0), (x, self.const.WINDOW_HEIGHT), (20, 20, 30), 1)
        for y in range(0, self.const.WINDOW_HEIGHT, 50):
            cv2.line(frame, (0, y), (self.const.WINDOW_WIDTH, y), (20, 20, 30), 1)
        return frame

    def draw_title(self, frame: np.ndarray) -> None:
        center_x = self.const.WINDOW_WIDTH // 2
        cv2.putText(frame, "VIRTUAL CHOIR CONDUCTOR", (center_x - 220, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 1)
        # cv2.putText(frame, "Learn how to conduct choir using this virtual training tool", (center_x - 170, 65),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 0), 1)

    def draw_pattern_guide(self, frame: np.ndarray, time_signature: str,
                           pattern: List[str], next_beat: str,
                           current_beat: int, show_completion: bool = False) -> np.ndarray:
        center_x = 280
        center_y = self.const.WINDOW_HEIGHT // 2 + 100
        size = self.const.PATTERN_SIZE

        if show_completion:
            cv2.putText(frame, "COMPLETED!", (center_x - 70, center_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
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
            cv2.putText(frame, beat_label, (point[0] - 50, point[1] - 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 200), 1)

        return frame

    def draw_satb_score(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        x = self.const.WINDOW_WIDTH - self.const.NOTATION_WIDTH - 20
        y = 90
        w = self.const.NOTATION_WIDTH
        h = self.const.NOTATION_HEIGHT

        score = None
        for s in ChoralLibrary.get_all():
            if s.id == stats['score_id']:
                score = s
                break

        if score:
            frame = self.notation_display.draw_satb_score(
                frame, score, stats['score_pos'],
                stats.get('upcoming_chords', []), x, y, w, h
            )

        return frame

    def draw_voice_controls(self, frame: np.ndarray, audio_gen: SATBAudioGenerator) -> np.ndarray:
        x = self.const.WINDOW_WIDTH - self.const.NOTATION_WIDTH - 20
        y = 90 + self.const.NOTATION_HEIGHT + 10
        w = self.const.NOTATION_WIDTH
        h = 170

        frame = self.notation_display.draw_voice_controls(frame, audio_gen, x, y, w, h)
        return frame

    def draw_info_panel(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 100), (15 + self.const.INFO_PANEL_WIDTH,
                                           100 + self.const.INFO_PANEL_HEIGHT),
                      (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        y_offset = 125

        if stats['show_completion_banner']:
            cv2.putText(frame, "CONGRATULATIONS!", (25, y_offset + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(frame, "You've completed the piece!", (25, y_offset + 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(frame, "Press [ and ] for next score", (25, y_offset + 105),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(frame, "or [r] to replay", (25, y_offset + 135),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 200, 0), 1)
            return frame

        if stats.get('show_resync_message', False):
            cv2.putText(frame, "RESYNC SUCCESSFUL!", (25, y_offset + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(frame, "Continue conducting from the next DOWNBEAT", (25, y_offset + 65),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
            return frame

        cv2.putText(frame, "STATUS", (25, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 0), 1)

        cv2.putText(frame, f"Score: {stats['score_id']} - {stats['score_name']}",
                    (25, y_offset + 35), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(frame, f"Composer: {stats['composer']}",
                    (25, y_offset + 60), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Time Signature: {stats['time_signature']} ({stats['beats_per_measure']} beats)",
                    (25, y_offset + 85), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1)

        if stats['has_tempo_error']:
            cv2.putText(frame, f"TEMPO: {stats['bpm']} / {stats['suggested_tempo']} BPM",
                        (25, y_offset + 115), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 150, 255), 1)
            cv2.putText(frame, stats['tempo_error_msg'],
                        (25, y_offset + 140), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 150, 255), 1)
            cv2.putText(frame, "Press 'a' to RESYNC",
                        (25, y_offset + 162), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 200, 255), 1)
        else:
            cv2.putText(frame, f"Tempo: {stats['bpm']} / {stats['suggested_tempo']} BPM",
                        (25, y_offset + 115), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"Beats: {stats['beats']}  |  Measures: {stats['measures']}",
                    (25, y_offset + 148), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Current Beat: {stats['current_beat']} / {stats['beats_per_measure']}",
                    (25, y_offset + 178), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 200, 100), 1)

        if stats.get('current_lyric'):
            cv2.putText(frame, f"Lyric: {stats['current_lyric']}",
                        (25, y_offset + 208), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 150, 150), 1)

        cv2.putText(frame, f"Progress: {stats['score_pos']} / {stats['score_total']} chords",
                    (25, y_offset + 240), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)

        if stats['beats'] > 0 and not stats['show_completion_banner']:
            cv2.putText(frame, f"FPS: {stats.get('fps', 0)}", (25, y_offset + 268),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def draw_keyboard_indicator(self, frame: np.ndarray, next_beat: str,
                                current_beat: int, beats_per_measure: int,
                                show_completion: bool = False, has_tempo_error: bool = False,
                                show_resync: bool = False) -> np.ndarray:
        x = self.const.WINDOW_WIDTH - self.const.SCORE_PANEL_WIDTH - self.const.SCORE_PANEL_X_OFFSET
        y = self.const.WINDOW_HEIGHT - self.const.KEYBOARD_PANEL_HEIGHT - self.const.SCORE_PANEL_X_OFFSET - 120

        cv2.rectangle(frame, (x, y), (self.const.WINDOW_WIDTH - 40,
                                      self.const.WINDOW_HEIGHT - 20),
                      (35, 35, 55), -1)
        cv2.rectangle(frame, (x, y), (self.const.WINDOW_WIDTH - 40,
                                      self.const.WINDOW_HEIGHT - 20),
                      (120, 120, 170), 2)

        if show_completion:
            cv2.putText(frame, "SCORE COMPLETE!", (x + 65, y + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 1)
            cv2.putText(frame, "Press [ and ] or [r]", (x + 70, y + 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 200, 0), 1)
            return frame

        if show_resync:
            cv2.putText(frame, "RESYNC SUCCESS!", (x + 65, y + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, "Start from DOWNBEAT", (x + 75, y + 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
            return frame

        if has_tempo_error:
            cv2.putText(frame, "TEMPO ERROR!", (x + 75, y + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 150, 255), 1)
            cv2.putText(frame, "Press 'a' to RESYNC", (x + 70, y + 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(frame, "(Keeps music position)", (x + 65, y + 90),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (200, 200, 200), 1)
            return frame

        next_beat_num = current_beat + 1 if current_beat < beats_per_measure else 1
        cv2.putText(frame, f"NEXT BEAT: {next_beat_num} / {beats_per_measure}",
                    (x + 65, y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 200, 0), 1)
        cv2.putText(frame, "PRESS:", (x + 35, y + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        key_map = {
            "DOWN": (120, 45, "DOWN"),
            "LEFT": (80, 70, "LEFT"),
            "RIGHT": (165, 70, "RIGHT"),
            "UP": (120, 45, "UP")
        }

        if next_beat in key_map:
            x_off, y_off, key_name = key_map[next_beat]
            kx = x + x_off
            ky = y + y_off
            cv2.rectangle(frame, (kx, ky), (kx + 80, ky + 30), (0, 100, 0), -1)
            cv2.rectangle(frame, (kx, ky), (kx + 80, ky + 30), (0, 255, 0), 2)
            cv2.putText(frame, key_name, (kx + 20, ky + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def draw_error(self, frame: np.ndarray, error_msg: str, error_time: float) -> np.ndarray:
        if error_msg and time.time() - error_time < 1.5:
            center_x = self.const.WINDOW_WIDTH // 2
            center_y = self.const.WINDOW_HEIGHT // 2 + 300
            (text_w, text_h), _ = cv2.getTextSize(error_msg, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame, (center_x - text_w // 2 - 15, center_y - text_h - 10),
                          (center_x + text_w // 2 + 15, center_y + 15),
                          (0, 0, 120), -1)
            cv2.rectangle(frame, (center_x - text_w // 2 - 15, center_y - text_h - 10),
                          (center_x + text_w // 2 + 15, center_y + 15),
                          (0, 0, 255), 2)
            cv2.putText(frame, error_msg, (center_x - text_w // 2, center_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1)
        return frame

    def draw_beat_flash(self, frame: np.ndarray, is_flash: bool, beat_name: str,
                        beat_number: int, beats_per_measure: int) -> np.ndarray:
        if is_flash:
            center_x = 280
            center_y = self.const.WINDOW_HEIGHT // 2 + 40
            overlay = frame.copy()
            cv2.circle(overlay, (center_x, center_y + 80), 70, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            text = f"BEAT {beat_number}/{beats_per_measure}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1)
            cv2.putText(frame, text, (center_x - text_w // 2, center_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)
        return frame

    def draw_controls(self, frame: np.ndarray, fps: int) -> np.ndarray:
        controls_text = "[m]Metronome  [r]RESET  [a]RESYNC  [:]PREV  []]NEXT  [s]Save  [q]Quit"
        (text_w, _), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.putText(frame, controls_text,
                    (self.const.WINDOW_WIDTH // 2 - text_w // 2, self.const.WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 200), 1)
        return frame


# ============================================
# MAIN APPLICATION
# ============================================

class SATBConductorApp:
    def __init__(self):
        self.conductor = SATBConductor()
        self.visualizer = SATBVisualizer()
        self._error_msg = ""
        self._error_time = 0
        self._fps = 0
        self._frame_count = 0
        self._fps_time = time.time()

        self._setup_window()
        self._print_welcome()

    def _setup_window(self) -> None:
        cv2.namedWindow('SATB Choir Conductor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SATB Choir Conductor',
                         VisualConstants.WINDOW_WIDTH,
                         VisualConstants.WINDOW_HEIGHT)

    def _print_welcome(self) -> None:
        print("=" * 70)
        print("SATB CHOIR CONDUCTOR")
        print("=" * 70)
        print("\nAvailable Choral Pieces:")
        for score in ChoralLibrary.get_all():
            chords = len(score.chords)
            print(f"  {score.id}. {score.name} ({score.composer}) - {score.time_signature} ({chords} chords)")
        print("\n[CONTROLS]")
        print("  Arrow Keys - Conduct (follow the pattern shown)")
        print("  [ - PREVIOUS piece")
        print("  ] - NEXT piece")
        print("  [m] - Metronome on/off")
        print("  [r] - RESET (restart score from beginning)")
        print("  [a] - RESYNC (fix tempo WITHOUT restarting)")
        print("  [s] - Save results")
        print("  [q] - Quit")
        print("\n[VOICE CONTROLS]")
        print("  Press 1-4 to mute/unmute voices:")
        print("    1 - Soprano   2 - Alto   3 - Tenor   4 - Bass")
        print("  Shift+1-4 to increase volume")
        print("\n[DISPLAY]")
        print("  Each voice displays notes on its own 5-line staff")
        print("  Soprano & Alto: Treble clef (G clef)")
        print("  Tenor & Bass: Bass clef (F clef)")
        print("=" * 70)

        stats = self.conductor.stats
        print(f"\nSTARTING WITH:")
        print(f"  Score {stats['score_id']}: {stats['score_name']} by {stats['composer']}")
        print(f"  Time Signature: {stats['time_signature']}")
        print(f"  Suggested Tempo: {stats['suggested_tempo']} BPM")
        print("=" * 70)

    def _process_input(self, key: int) -> bool:
        # Voice mute keys (1-4)
        if 49 <= key <= 52:
            voice_idx = key - 49
            self.conductor.mute_voice(voice_idx)
            return True

        # Shift+number for volume up
        if key == 33:  # Shift+1
            self.conductor.adjust_voice_volume(0, 0.05)
            return True
        elif key == 64:  # Shift+2
            self.conductor.adjust_voice_volume(1, 0.05)
            return True
        elif key == 35:  # Shift+3
            self.conductor.adjust_voice_volume(2, 0.05)
            return True
        elif key == 36:  # Shift+4
            self.conductor.adjust_voice_volume(3, 0.05)
            return True

        # Conducting beats
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
        elif key == ord('a'):
            self.conductor.resync_tempo()
            self._error_msg = "RESYNC - Continue from DOWNBEAT"
            self._error_time = time.time()
        elif key == ord('['):  # Left bracket - Previous piece
            self.conductor.previous_score()
            self._error_msg = "Switched to previous piece"
            self._error_time = time.time()
        elif key == ord(']'):  # Right bracket - Next piece
            self.conductor.next_score()
            self._error_msg = "Switched to next piece"
            self._error_time = time.time()
        elif key == ord('m'):
            self.conductor.toggle_metronome()
        elif key == ord('s'):
            self._save_results()

        return True

    def _save_results(self) -> None:
        stats = self.conductor.stats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"satb_conductor_results_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write("SATB CHOIR CONDUCTOR - SESSION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Score: {stats['score_id']} - {stats['score_name']}\n")
            f.write(f"Composer: {stats['composer']}\n")
            f.write(f"Time Signature: {stats['time_signature']}\n")
            f.write(f"Pattern: {stats['pattern_display']}\n")
            f.write(f"Suggested Tempo: {stats['suggested_tempo']} BPM\n")
            f.write(f"Average BPM: {stats['bpm']}\n")
            f.write(f"Total Beats: {stats['beats']}\n")
            f.write(f"Total Measures: {stats['measures']}\n")
            f.write(f"Progress: {stats['score_pos']}/{stats['score_total']} chords\n")
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
        frame = self.visualizer.draw_satb_score(frame, stats_with_fps)
        frame = self.visualizer.draw_voice_controls(frame, self.conductor._audio)
        frame = self.visualizer.draw_info_panel(frame, stats_with_fps)
        frame = self.visualizer.draw_keyboard_indicator(
            frame, stats['next_beat'],
            stats['current_beat'],
            stats['beats_per_measure'],
            stats['show_completion_banner'],
            stats['has_tempo_error'],
            stats['show_resync_message']
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
        print("Thank you for conducting the choir!")

    def run(self) -> None:
        while True:
            frame = self._create_frame()
            cv2.imshow('SATB Choir Conductor', frame)
            key = cv2.waitKey(50) & 0xFF

            if key == 0:
                continue

            if not self._process_input(key):
                break

        cv2.destroyAllWindows()
        self._print_summary()


def main() -> None:
    app = SATBConductorApp()
    app.run()


if __name__ == "__main__":
    main()