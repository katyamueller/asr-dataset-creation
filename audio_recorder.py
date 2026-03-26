
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import csv
import os
import json
import time
from datetime import datetime
from threading import Thread
import queue

# --- CONFIGURATION ---
SAMPLE_RATE = 16000  # Whisper standard
DURATION_LIMIT = 15  # Max seconds per recording (safety cutoff)
OUTPUT_DIR = "recordings"
MANIFEST_FILE = "dataset_manifest.jsonl"
PHRASES_FILE = "phrases.csv"

class ASRRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("ASR Data Collection Tool")
        self.root.geometry("800x600")
        
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
        
        # Ensure output dir exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        # Load phrases
        self.load_phrases()
        
        # UI Setup
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
            for i, row in enumerate(reader):
                if not row: continue
                # Handle both "just text" and "id,text" formats
                if len(row) == 1:
                    temp_phrases.append({"id": f"seg_{i:05d}", "text": row[0]})
                else:
                    temp_phrases.append({"id": row[0], "text": row[1]})
        
        # Check existing files in output directory
        existing_files = set(os.listdir(OUTPUT_DIR))

        # Filter out completed items
        self.phrases_to_record = []
        skipped_count = 0
        
        for item in temp_phrases:
            filename = f"{item['id']}.wav"
            if filename in existing_files:
                skipped_count += 1
            else:
                self.phrases_to_record.append(item)
        
        self.all_phrases = temp_phrases # Keep reference if needed
        
        msg = f"Loaded {len(temp_phrases)} total phrases.\nFound {skipped_count} existing recordings.\n{len(self.phrases_to_record)} segments remaining."
        print(msg)

        ## Show a small info box on startup if many were skipped
        # if skipped_count > 0:
        #     self.root.after(500, lambda: messagebox.showinfo("Resume Detected", msg))

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

        self.lbl_device_info = ttk.Label(device_frame, text="", font=("Arial", 9))
        self.lbl_device_info.pack(side=tk.LEFT, padx=10)

        # Second Frame: Progress
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        self.lbl_progress = ttk.Label(top_frame, text="Progress: 0 / 0", font=("Arial", 12, "bold"))
        self.lbl_progress.pack(side=tk.LEFT)
        
        self.btn_save_manifest = ttk.Button(top_frame, text="Save Data & Exit", command=self.save_manifest_and_exit)
        self.btn_save_manifest.pack(side=tk.RIGHT)

        # Middle Frame: Prompt Display
        mid_frame = ttk.Frame(self.root, padding="20")
        mid_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_phrases = ttk.Label(mid_frame, text="", wraplength=700, font=("Arial", 18, "bold"), justify=tk.CENTER)
        self.lbl_phrases.pack(expand=True)
        
        self.lbl_status = ttk.Label(mid_frame, text="Press SPACE to Record", foreground="gray", font=("Arial", 12))
        self.lbl_status.pack(pady=10)

        # Bottom Frame: Controls
        bot_frame = ttk.Frame(self.root, padding="10")
        bot_frame.pack(fill=tk.X)
        
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

    def toggle_record(self, event=None):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.lbl_status.config(text="🔴 RECORDING... (Press Space to Stop)", foreground="red")
        self.btn_record.config(text="Stop (Space)")
        
        try:
            self.stream = sd.InputStream(device=self.selected_device, samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                         callback=self.audio_callback)
            self.stream.start()
            self.start_time = time.time()
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not start recording:\n{e}")
            self.is_recording = False
            self.lbl_status.config(text="Error starting audio", foreground="red")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback to collect audio chunks and compute levels."""
        if status:
            print(status)
        
        self.audio_data.append(indata.copy())

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        self.stream=None
        
        duration = time.time() - self.start_time
        self.lbl_status.config(text=f"Recorded {duration:.2f}s. Press Enter to Save or Space to Retry.", foreground="blue")
        self.btn_record.config(text="Record (Space)")
        self.btn_next.state(['!disabled'])
        self.btn_retry.state(['!disabled'])

    def save_current_audio(self):
        if not self.audio_data:
            return None
            
        recording = np.concatenate(self.audio_data, axis=0)
        recording_int16 = (recording * 32767).astype(np.int16)
        
        current_phrase = self.phrases_to_record[self.current_index]
        filename = f"{current_phrase['id']}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save WAV
        write(filepath, SAMPLE_RATE, recording_int16)
        
        return {
            "audio_filepath": os.path.abspath(filepath),
            "text": current_phrase['text'],
            "id": current_phrase['id'],
            "duration": len(recording) / SAMPLE_RATE
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
        
        all_entries = []
        
        # 1. Add newly recorded in this session
        all_entries.extend(self.saved_files)
        
        # 2. Scan disk for previous sessions' files that weren't in this session's list
        # (This handles cases where you ran the script twice)
        existing_ids_in_session = {e['id'] for e in self.saved_files}
        
        for item in self.all_phrases:
            if item['id'] in existing_ids_in_session:
                continue # Already added from current session
            
            filename = f"{item['id']}.wav"
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