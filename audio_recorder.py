
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import resample_poly
from math import gcd
import csv
import os
import json
import time
from datetime import datetime
from threading import Thread
import queue

import matplotlib
matplotlib.use('TkAgg')  # Ensure it uses the Tk backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- CONFIGURATION ---
TARGET_SAMPLE_RATE = 16000 
TRIM_START_MS = 200 
OUTPUT_DIR = "recordings"
MANIFEST_FILE = "dataset_manifest.jsonl"
PHRASES_FILE = "phrases.csv"
CONFIG_FILE = "recorder_config.json"
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0
MIN_RMS_THRESHOLD = 0.001
CLIP_FRACTION_THRESHOLD = 0.01  # 1%
CLIP_WARNING_COUNT = 10

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config: dict):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def _downsample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate:
        return audio
    g = gcd(orig_rate, target_rate)
    return resample_poly(audio, target_rate // g, orig_rate // g)

def ask_speaker_name(root) -> str:
    while True:
        name = simpledialog.askstring(
            "Speaker Name",
            "Enter your first name:",
            parent=root
        )
        if name is None:
            root.destroy()
            raise SystemExit("No speaker name provided. Exiting.")
        name = name.strip().replace(" ", "_")
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if name and all(c in safe_chars for c in name):
            return name
        messagebox.showwarning("Invalid Name",
            "Please use only letters or digits.", parent=root)

class ASRRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("ASR Data Collection Tool")
        config = load_config()
        if "speaker_name" in config:
            self.speaker_name = config["speaker_name"]
        else:
            self.speaker_name = ask_speaker_name(self.root)
            config["speaker_name"] = self.speaker_name
            save_config(config)
        self.root.geometry("800x600")
        self.clipped_recording_count = 0
        
        # State variables
        self.all_phrases = []
        self.phrases_to_record = []
        self.current_index = 0
        self.is_recording = False
        self.audio_data = []
        self.saved_files = []

        self.selected_device = None
        self.stream = None

        self.combo_device = None

        # Audio level queue for visualization
        self.level_queue = queue.Queue()
        
        # Visualization variables
        self.visualizer_active = False
        self.num_bars = 10  # Number of bars for visualization
        self.bars = []

        self.bar_levels = np.zeros(self.num_bars)

        self.fig = None
        self.canvas = None
        self.ax = None
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        self.load_phrases()
        
        self.setup_ui()

        self.populate_devices()
        
        # Bind keys
        self.root.bind('<space>', self.toggle_record)
        self.root.bind('<Return>', self.next_item)
        self.root.bind('<r>', lambda e: self.retry_last())
        self.root.bind('<Escape>', lambda e: self.root.quit())

        # Handle case where everything is already done
        if not self.phrases_to_record:
            self.lbl_phrases.config(text="✅ All recordings complete!\nCheck your manifest file.")
            self.lbl_status.config(text="No pending segments found.")
            self.btn_record.state(['disabled'])
            self.btn_next.state(['disabled'])
        else:
            self.update_display()

    def get_audio_devices(self):
        devices = sd.query_devices()
        default_idx = sd.query_devices(kind='input')['name']
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                name = device['name']
                if name == default_idx:
                    input_devices.insert(0, (i, f"{name} (default)"))
                else: input_devices.append((i, name))

        return input_devices

    def populate_devices(self):
        """Populate device dropdown with available audio devices."""
        devices = self.get_audio_devices()
        
        if not devices:
            messagebox.showerror("Error", "No audio input devices found!")
            self.root.quit()
            return
        
        device_names = [f"{name}" for _, name in devices]
        self.selected_device = device_names[0]
        self.combo_device['values'] = device_names
        self.device_map = {name: device_id for device_id, name in devices}
        
        # Set default device
        self.selected_device = devices[0][0]

        self.combo_device.set(device_names[0])

    def on_device_change(self, event=None):
        """Handle device selection change."""
        selected_name = self.combo_device.get()
        self.selected_device = self.device_map[selected_name]

    def make_filename(self, phrase_id: str) -> str:
        return f"{self.speaker_name}_{phrase_id}.wav"

    def load_phrases(self):
        """Load prompts and remove those that already have audio files."""
        if not os.path.exists(PHRASES_FILE):
            messagebox.showerror("Error", f"File '{PHRASES_FILE}' not found.")

            with open(PHRASES_FILE, 'w', encoding='utf-8') as f:
                f.write("The quick brown fox jumps over the lazy dog.\n")
                f.write("Speech recognition models require diverse data.\n")
                f.write("Add your own data in the phrases.csv file.\n")
            messagebox.showinfo("Info", f"Created example '{PHRASES_FILE}'. Please edit it with your custom phrases, then restart.")
            
            self.root.quit()
            return

        temp_phrases = []
        with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # for i, row in enumerate(reader):
            #     if not row: continue
            #     # Handle both "just text" and "id,text" formats
            #     if len(row) == 1:
            #         temp_phrases.append({"id": f"seg_{i:05d}", "text": row[0]})
            #     else:
            #         temp_phrases.append({"id": row[0], "text": row[1]})

            for i, row in enumerate(reader):
                if not row:
                    continue
                if len(row) == 1:
                    col = row[0].strip()
                    if i == 0 and col.lower() in ('text', 'phrase', 'sentence'):
                        continue  # skip header
                    temp_phrases.append({"id": f"seg_{i:05d}", "text": col})
                else:
                    id_col, text_col = row[0].strip(), row[1].strip()
                    if i == 0 and id_col.lower() == 'id':
                        continue  # skip header
                    if id_col and text_col:
                        temp_phrases.append({"id": id_col, "text": text_col})
        
        # Check existing files in output directory
        existing_files = set(os.listdir(OUTPUT_DIR))

        # Filter out completed items
        self.phrases_to_record = []
        skipped_count = 0
        
        for item in temp_phrases:
            filename = self.make_filename(item['id'])
            if filename in existing_files:
                skipped_count += 1
            else:
                self.phrases_to_record.append(item)
        
        self.all_phrases = temp_phrases # Keep reference if needed
        
        msg = f"Loaded {len(temp_phrases)} total phrases.\nFound {skipped_count} existing recordings.\n{len(self.phrases_to_record)} segments remaining."
        print(msg)

    def setup_ui(self):
        
        # ===== TOP FRAME: Device Selection =====
        device_frame = ttk.LabelFrame(self.root, padding="10")
        device_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(device_frame, text="Select Input Device:").pack(side=tk.LEFT, padx=5)

        self.selected_device = tk.StringVar()
        self.combo_device = ttk.Combobox(device_frame, textvariable=self.selected_device, 
                                 state='readonly', width=50)
        self.combo_device.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.combo_device.bind('<<ComboboxSelected>>', self.on_device_change)

        ttk.Label(
            device_frame,
            text=f"👤  {self.speaker_name}",
            font=("Arial", 10, "bold"),
            foreground="#2255aa",
            relief="solid",
            padding=(6, 2),
        ).pack(side=tk.RIGHT, padx=8)

        self.lbl_device_info = ttk.Label(device_frame, text="", font=("Arial", 9))
        self.lbl_device_info.pack(side=tk.LEFT, padx=10)

        # Second Frame: Progress
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        self.lbl_progress = ttk.Label(top_frame, text="Progress: 0 / 0", font=("Arial", 12, "bold"))
        self.lbl_progress.pack(side=tk.LEFT)
        
        self.btn_save_manifest = ttk.Button(top_frame, text="Save Data & Exit", command=self.save_manifest_and_exit)
        self.btn_save_manifest.pack(side=tk.RIGHT)

        # Middle Frame: Phrase Display
        mid_frame = ttk.Frame(self.root, padding="20")
        mid_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_phrases = ttk.Label(mid_frame, text="", wraplength=700, font=("Arial", 18, "bold"), justify=tk.CENTER)
        self.lbl_phrases.pack(expand=True)
        
        self.lbl_status = ttk.Label(mid_frame, text="Press SPACE to Record", foreground="gray", font=("Arial", 12))
        self.lbl_status.pack(pady=10)

        # === Visualizer Frame ===
        vis_frame = ttk.Frame(mid_frame, padding="5")
        vis_frame.pack(pady=10)
        
        self.lbl_vis_status = ttk.Label(vis_frame, text="Microphone: Off", foreground="gray", font=("Arial", 9))
        self.lbl_vis_status.pack(pady=5)

        tk_bg = self.root.cget("bg")  # Tk-compatible color string
        rgb = self.root.winfo_rgb(tk_bg)
        mpl_bg = tuple(v / 65535 for v in rgb) # matplotlib compatible color tuple

        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.fig.patch.set_facecolor(mpl_bg)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(mpl_bg)
        self.ax.axis('off')
        self.ax.set_xlim(0, self.num_bars)
        self.ax.set_ylim(0, 1.0)
        
        # Create 10 bars for spectogram-ish audio visualization
        self.bars = []
        for i in range(self.num_bars):
            bar = self.ax.bar(i, 0, width=0.6, color='#808080', align='center')
            self.bars.append(bar[0])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)

        widget = self.canvas.get_tk_widget()
        widget.configure(bg=tk_bg, highlightthickness=0)
        widget.pack()

        # Bottom Frame: Controls
        bot_frame = ttk.Frame(self.root, padding="10")
        bot_frame.pack(fill=tk.X)

        self.btn_listen = ttk.Button(bot_frame, text="Listen Back", command=self.listen_back)
        self.btn_listen.pack(side=tk.LEFT, padx=5)
        self.btn_listen.state(['disabled'])
        
        self.btn_record = ttk.Button(bot_frame, text="Record (Space)", command=self.toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=5)
        
        self.btn_retry = ttk.Button(bot_frame, text="Retry Last (R)", command=self.retry_last)
        self.btn_retry.pack(side=tk.LEFT, padx=5)
        self.btn_retry.state(['disabled'])
        
        self.btn_next = ttk.Button(bot_frame, text="Next/Save (Enter)", command=self.next_item)
        self.btn_next.pack(side=tk.RIGHT, padx=5)
        self.btn_next.state(['disabled'])

    def update_display(self):
        if self.current_index < len(self.phrases_to_record):

            data = self.phrases_to_record[self.current_index]
            remaining = len(self.phrases_to_record) - self.current_index
            total_original = len(self.all_phrases)

            self.lbl_phrases.config(text=f"[{data['id']}]\n{data['text']}")
            self.lbl_progress.config(text=f"Remaining: {remaining} | Total Done: {total_original - remaining}")
            self.lbl_status.config(text="Press SPACE to Record", foreground="gray")
        
        else:
            self.lbl_phrases.config(text="🎉 All recordings complete!")
            self.lbl_status.config(text="Please save data and exit.")
            self.btn_record.state(['disabled'])
            self.btn_next.state(['disabled'])

    def listen_back(self):
        if not self.audio_data:
            return

        recording = np.concatenate(self.audio_data, axis=0)
        
        # Apply the same processing pipeline as save_current_audio
        trim_samples = int((TRIM_START_MS / 1000) * self.actual_sample_rate)
        recording = recording[trim_samples:]
        recording = _downsample(recording, self.actual_sample_rate, TARGET_SAMPLE_RATE)
        recording = np.clip(recording, -1.0, 1.0)

        self.lbl_status.config(text="▶️ Playing back...", foreground="gray")
        self.btn_listen.state(['disabled'])
        self.root.update()

        def _play():
            sd.play(recording, samplerate=TARGET_SAMPLE_RATE)
            sd.wait()
            self.root.after(0, lambda: self.btn_listen.state(['!disabled']))
            self.root.after(0, lambda: self.lbl_status.config(
                text="Listen back, then press Enter to Save or Space to Retry.",
                foreground="blue"))

        Thread(target=_play, daemon=True).start()

    def toggle_record(self, event=None):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.start_visualizer()
        self.lbl_status.config(text="🔴 RECORDING... (Press Space to Stop)", foreground="red")
        self.btn_record.config(text="Stop (Space)")
        
        try:
            self.stream = sd.InputStream(device=self.selected_device, channels=1, dtype='float32',
                                         callback=self.audio_callback)
            self.stream.start()
            self.start_time = time.time()
            self.actual_sample_rate = int(self.stream.samplerate)

        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not start recording:\n{e}")
            self.is_recording = False
            self.lbl_status.config(text="Error starting audio", foreground="red")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback to collect audio chunks and compute levels."""
        if status:
            print(status)
        
        self.audio_data.append(indata.copy())
        
        if indata.size > 0:
            rms = np.sqrt(np.mean(indata**2))
            
            # Scaling logic (adjust threshold as needed)
            max_threshold = 0.05 
            if rms < 0.001:
                rms = 0.0
            else:
                rms = min(rms / max_threshold, 1.0)
            
            # Add a tiny bit of random noise to each bar for "live" feel
            # This prevents them from all moving in perfect lockstep
            self.level_queue.put(rms)

    def update_visualizer(self):
        """Update the 10 bars based on audio level."""
        if not self.visualizer_active:
            return

        while not self.level_queue.empty():
            try:
                base_level = self.level_queue.get_nowait()
                
                # Animate bars according to audio levels, add time delay and sin for smoother animation, middle bars higher
                decay = 0.9  # to make it less jumpy, closer to 1 = slower fall

                for i, bar in enumerate(self.bars):
                    self.bar_levels[i] *= decay
                    variation = 0.8 + 0.2 * np.sin(time.time() * 3 + i)

                    center = self.num_bars / 2
                    distance = abs(i - center) / center
                    shape = 1 - distance  # center = 1, edges = 0

                    target = base_level * variation * shape

                    target = min(target, 1.0)

                    # Smooth transition (EMA)
                    alpha = 0.2  # smaller = smoother, larger = more reactive
                    self.bar_levels[i] = (1 - alpha) * self.bar_levels[i] + alpha * target

                    bar.set_height(self.bar_levels[i])
                
            except:
                break
        
        # Redraw the canvas
        self.canvas.draw_idle()
        
        # Schedule next update
        if self.visualizer_active:
            self.root.after(40, self.update_visualizer)

    def start_visualizer(self):
        self.visualizer_active = True
        self.lbl_vis_status.config(text="Microphone: Active", foreground="green")
        # Reset bars to 0 just in case
        for bar in self.bars:
            bar.set_height(0)
        self.canvas.draw_idle()
        self.update_visualizer()

    def stop_visualizer(self):
        self.visualizer_active = False
        self.lbl_vis_status.config(text="Microphone: Off", foreground="gray")
        for bar in self.bars:
            bar.set_height(0)
        self.canvas.draw_idle()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.stop_visualizer()

        duration = time.time() - self.start_time

        if duration < MIN_DURATION_S:
            self.audio_data = []
            self.lbl_status.config(text=f"⚠️ Too short ({duration:.2f}s). Please re-record.", foreground="orange")
            self.btn_record.config(text="Record (Space)")
            return

        if duration > MAX_DURATION_S:
            self.audio_data = []
            self.lbl_status.config(text=f"⚠️ Too long ({duration:.2f}s, max {MAX_DURATION_S}s). Please re-record.", foreground="orange")
            self.btn_record.config(text="Record (Space)")
            return

        self.lbl_status.config(
            text=f"Recorded {duration:.2f}s. Listen back, then press Enter to Save or Space to Retry.",
            foreground="white",
        )
        self.btn_record.config(text="Record (Space)")
        self.btn_next.state(['!disabled'])
        self.btn_retry.state(['!disabled'])
        self.btn_listen.state(['!disabled'])  # enable listen button

    def save_current_audio(self):
        if not self.audio_data:
            return None

        recording = np.concatenate(self.audio_data, axis=0)

        # RMS check
        rms = np.sqrt(np.mean(recording**2))
        if rms < MIN_RMS_THRESHOLD:
            messagebox.showwarning("Silent Recording",
                "The recording appears to be silent or very quiet.\n"
                "Check your microphone and try again.")
            self.audio_data = []
            self.btn_next.state(['disabled'])
            self.btn_listen.state(['disabled'])
            return None

        # Clip check
        clipped_samples = np.sum(np.abs(recording) >= 1.0)
        clip_fraction = clipped_samples / len(recording)
        if clip_fraction > CLIP_FRACTION_THRESHOLD:
            self.clipped_recording_count += 1
            print(f"[WARN] Clipping detected: {clip_fraction:.1%} of samples clipped "
                f"(recording #{self.clipped_recording_count})")
            if self.clipped_recording_count >= CLIP_WARNING_COUNT:
                messagebox.showwarning("Persistent Clipping Detected",
                    f"{self.clipped_recording_count} recordings have had significant clipping.\n"
                    "Please check the quality of your recordings and/ or reach out to us.")
                self.clipped_recording_count = 0  # reset so it doesn't fire every recording

        # space bar click was audible in recordings, so added delay 
        trim_samples = int((TRIM_START_MS / 1000) * self.actual_sample_rate)
        recording = recording[trim_samples:]

        # downsample to 16kHz (sample rate used by ASR models)
        recording = _downsample(recording, self.actual_sample_rate, TARGET_SAMPLE_RATE)
        recording = np.clip(recording, -1.0, 1.0)

        # Normalize to use the full dynamic range
        max_val = np.abs(recording).max()
        if max_val > 0:
            recording = recording / max_val * 0.95 

        # convert to 16bit int
        recording_int16 = (recording * 32767).astype(np.int16) 

        current_phrase = self.phrases_to_record[self.current_index]
        filename = self.make_filename(current_phrase['id'])
        filepath = os.path.join(OUTPUT_DIR, filename)

        # save as tmp first in case of curruption while saving 
        tmp_path = filepath + ".tmp"
        write(tmp_path, TARGET_SAMPLE_RATE, recording_int16)
        os.rename(tmp_path, filepath)

        return {
            "audio_filepath": os.path.abspath(filepath),
            "text": current_phrase['text'],
            "id": current_phrase['id'],
            "speaker": self.speaker_name,
            "duration": len(recording) / TARGET_SAMPLE_RATE,
            "sample_rate": TARGET_SAMPLE_RATE,
            "timestamp": datetime.now().isoformat(),
        }

    def next_item(self, event=None):
        if not self.is_recording and self.audio_data:
            # Save and move next
            entry = self.save_current_audio()
            if entry:
                self.saved_files.append(entry)
                self.current_index += 1
                self.audio_data = []
                self.btn_next.state(['disabled'])
                self.btn_retry.state(['disabled'])
                self.lbl_status.config(text="Press SPACE to Record", foreground="gray")
                self.update_display()
        elif self.current_index < len(self.phrases_to_record) and not self.audio_data:
            # Just skip (no recording made)
            self.current_index += 1
            self.update_display()

    def retry_last(self, event=None):
        if not self.is_recording and self.audio_data:
            self.audio_data = []
            self.lbl_status.config(text="Retrying... Press SPACE to Record", foreground="gray")
            self.btn_next.state(['disabled'])
            self.btn_retry.state(['disabled'])

    def save_manifest_and_exit(self):
        # merge newly recorded files with any existing ones in the folder

        entry = self.save_current_audio()
        if entry:
            self.saved_files.append(entry)
            self.current_index += 1
            self.audio_data = []
            self.btn_next.state(['disabled'])
            self.btn_retry.state(['disabled'])
        
        all_entries = []
        
        # 1. Add newly recorded in this session
        all_entries.extend(self.saved_files)
        
        # 2. Scan disk for previous sessions' files that weren't in this session's list
        # (This handles cases where you ran the script twice)
        existing_ids_in_session = {e['id'] for e in self.saved_files}
        
        for item in self.all_phrases:
            if item['id'] in existing_ids_in_session:
                continue # Already added from current session
            
            filename = self.make_filename(item['id'])
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                # Calculate duration for existing file
                try:
                    rate, data = read(filepath)
                    duration = float(len(data) / rate)
                except:
                    duration = 0.0
                
                all_entries.append({
                    "audio_filepath": os.path.abspath(filepath),
                    "text": item['text'],
                    "id": item['id'],
                    "duration": duration
                })

        # Sort by ID to keep manifest organized
        all_entries.sort(key=lambda x: x['id'])

        with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
                
        messagebox.showinfo("Success", f"Manifest saved to {MANIFEST_FILE}\nTotal segments: {len(all_entries)}")
        self.root.quit()

    def on_close(self, event=None):
        if self.is_recording:
            self.stop_recording()   

        if messagebox.askokcancel("Quit", "Stop recording? Progress is saved automatically on disk.\nYou can resume later."):
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASRRecorder(root)
    root.mainloop()