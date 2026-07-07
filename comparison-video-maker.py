import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import re
import os
from pathlib import Path
from datetime import datetime

from render_comparison import render_comparison_video

BASE_DIR   = Path(__file__).parent
CHAR_DIR   = BASE_DIR / "assets" / "character"
TTS_DIR    = BASE_DIR / "assets" / "tts"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPTS_DIR = BASE_DIR / "assets" / "prompt"

MASTER_PROMPT_PATH = PROMPTS_DIR / "master prompt-comparizon.txt"
MASTER_PROMPT = MASTER_PROMPT_PATH.read_text(encoding="utf-8")

MUSIC_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".aac"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def _scan_assets(folder, exts):
    try:
        return sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])
    except Exception:
        return []

def _extract_json(text):
    """Extract JSON array from text that may contain other content (image prompts)."""
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in input")
    end = text.rfind("]")
    if end == -1:
        raise ValueError("No closing bracket found in input")
    raw = text[start:end+1]
    return json.loads(raw)

def _extract_image_prompts(text):
    """Extract the IMAGE PROMPTS section for display, return raw text."""
    lines = text.split("\n")
    in_prompts = False
    prompts_lines = []
    for line in lines:
        if line.strip().startswith("=== IMAGE PROMPTS ==="):
            in_prompts = True
            continue
        if line.strip().startswith("=== SCRIPT ==="):
            break
        if in_prompts and line.strip():
            prompts_lines.append(line)
    return "\n".join(prompts_lines) if prompts_lines else "(No image prompts section found)"


class ComparisonVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparison Video Maker")
        self.root.geometry("750x820")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        self.status_var   = tk.StringVar(value="Ready.")
        self.progress_var = tk.DoubleVar(value=0)
        self.cancel_event = threading.Event()
        self.image_paths  = [tk.StringVar() for _ in range(6)]
        self.bg_music_var = tk.StringVar()
        self._build_ui()
        self._refresh_asset_status()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#16213e", background="#e94560", thickness=8)

        # Title
        tk.Label(self.root, text="COMPARISON VIDEO MAKER",
                 font=("Consolas", 14, "bold"),
                 bg="#1a1a2e", fg="#e94560").pack(pady=(14, 2))
        tk.Label(self.root, text="TikTok / YT Shorts  •  1080×1920  •  3 pairs",
                 font=("Consolas", 9), bg="#1a1a2e", fg="#666").pack(pady=(0, 6))

        # Copy Master Prompt button
        self.copy_btn = tk.Button(
            self.root, text="COPY MASTER PROMPT",
            command=self._copy_master_prompt,
            bg="#0f3460", fg="#a8d8ea",
            font=("Consolas", 9, "bold"),
            relief="flat", cursor="hand2", padx=14, pady=5
        )
        self.copy_btn.pack(pady=(0, 8))

        # Input area
        input_frame = tk.LabelFrame(self.root, text=" Paste LLM Output Here ",
                                    font=("Consolas", 9),
                                    bg="#16213e", fg="#aaa",
                                    bd=1, relief="flat", padx=10, pady=6)
        input_frame.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        in_scroll = tk.Scrollbar(input_frame)
        in_scroll.pack(side="right", fill="y")

        self.input_text = tk.Text(
            input_frame, height=6,
            bg="#0d0d1a", fg="#a8d8a8",
            font=("Consolas", 9),
            insertbackground="#e94560",
            bd=0, highlightthickness=0,
            yscrollcommand=in_scroll.set
        )
        self.input_text.pack(fill="both", expand=True)
        in_scroll.config(command=self.input_text.yview)

        # Image prompts display
        prompts_frame = tk.LabelFrame(self.root, text=" Image Prompts (read-only) ",
                                      font=("Consolas", 9),
                                      bg="#16213e", fg="#aaa",
                                      bd=1, relief="flat", padx=10, pady=6)
        prompts_frame.pack(fill="x", padx=16, pady=(0, 6))

        p_scroll = tk.Scrollbar(prompts_frame)
        p_scroll.pack(side="right", fill="y")

        self.prompts_text = tk.Text(
            prompts_frame, height=4,
            bg="#0d0d1a", fg="#7fcc7f",
            font=("Consolas", 8),
            bd=0, highlightthickness=0,
            state="disabled",
            yscrollcommand=p_scroll.set
        )
        self.prompts_text.pack(fill="both", expand=True)
        p_scroll.config(command=self.prompts_text.yview)

        # Image selectors
        img_frame = tk.LabelFrame(self.root, text=" Item Images (2 per pair × 3 pairs) ",
                                  font=("Consolas", 9),
                                  bg="#16213e", fg="#aaa",
                                  bd=1, relief="flat", padx=10, pady=6)
        img_frame.pack(fill="x", padx=16, pady=(0, 6))

        labels = [
            "Pair 1 — Image A:", "Pair 1 — Image B:",
            "Pair 2 — Image A:", "Pair 2 — Image B:",
            "Pair 3 — Image A:", "Pair 3 — Image B:",
        ]
        for i, lbl in enumerate(labels):
            row = tk.Frame(img_frame, bg="#16213e")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, font=("Consolas", 8),
                     bg="#16213e", fg="#ccc", width=18, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=self.image_paths[i],
                     font=("Consolas", 8), bg="#0d0d1a", fg="#a8d8ea",
                     bd=0, relief="flat").pack(side="left", fill="x", expand=True, padx=(0, 4))
            tk.Button(row, text="Browse", font=("Consolas", 8),
                      bg="#0f3460", fg="#a8d8ea",
                      relief="flat", cursor="hand2", padx=8,
                      command=lambda idx=i: self._browse_image(idx)).pack(side="right")

        # BG Music
        music_frame = tk.LabelFrame(self.root, text=" Background Music (optional) ",
                                    font=("Consolas", 9),
                                    bg="#16213e", fg="#aaa",
                                    bd=1, relief="flat", padx=10, pady=6)
        music_frame.pack(fill="x", padx=16, pady=(0, 6))

        mrow = tk.Frame(music_frame, bg="#16213e")
        mrow.pack(fill="x")
        tk.Entry(mrow, textvariable=self.bg_music_var,
                 font=("Consolas", 8), bg="#0d0d1a", fg="#a8d8ea",
                 bd=0, relief="flat").pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(mrow, text="Browse", font=("Consolas", 8),
                  bg="#0f3460", fg="#a8d8ea",
                  relief="flat", cursor="hand2", padx=8,
                  command=self._browse_music).pack(side="right")

        # Asset status
        self.status_label = tk.Label(self.root, text="", font=("Consolas", 8),
                                     bg="#1a1a2e", fg="#7fcc7f")
        self.status_label.pack(fill="x", padx=16, pady=(0, 2))

        # Progress + status
        prog_frame = tk.Frame(self.root, bg="#1a1a2e")
        prog_frame.pack(fill="x", padx=16, pady=(0, 2))
        ttk.Progressbar(prog_frame, variable=self.progress_var,
                        maximum=100, style="TProgressbar").pack(fill="x")

        tk.Label(self.root, textvariable=self.status_var,
                 font=("Consolas", 9), bg="#1a1a2e", fg="#888").pack(pady=(0, 2))

        # Log
        log_frame = tk.LabelFrame(self.root, text=" Log ",
                                  font=("Consolas", 9),
                                  bg="#16213e", fg="#aaa",
                                  bd=1, relief="flat", padx=10, pady=4)
        log_frame.pack(fill="x", padx=16, pady=(0, 6))

        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")

        self.log_box = tk.Text(log_frame, height=5,
                               bg="#0d0d1a", fg="#7fcc7f",
                               font=("Consolas", 8),
                               bd=0, highlightthickness=0,
                               state="disabled",
                               yscrollcommand=log_scroll.set)
        self.log_box.pack(fill="both", expand=True)
        log_scroll.config(command=self.log_box.yview)

        # Generate button
        self.gen_btn = tk.Button(
            self.root, text="GENERATE VIDEO",
            command=self._toggle_generation,
            bg="#e94560", fg="white",
            font=("Consolas", 12, "bold"),
            relief="flat", cursor="hand2",
            padx=20, pady=10
        )
        self.gen_btn.pack(pady=(0, 12))

        # Bind input change to extract prompts
        self.input_text.bind("<KeyRelease>", self._on_input_change)

    def _browse_image(self, idx):
        path = filedialog.askopenfilename(
            title=f"Select image for pair {idx//2 + 1}",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")]
        )
        if path:
            self.image_paths[idx].set(path)

    def _browse_music(self):
        path = filedialog.askopenfilename(
            title="Select background music",
            filetypes=[("Audio", "*.mp3 *.wav *.m4a *.ogg"), ("All files", "*.*")]
        )
        if path:
            self.bg_music_var.set(path)

    def _refresh_asset_status(self):
        chars = _scan_assets(CHAR_DIR, {".mp4"})
        self.status_label.config(text=f"Character videos: {len(chars)}  |  TTS models ready")

    def _copy_master_prompt(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(MASTER_PROMPT)
        self.root.update()
        self.copy_btn.config(text="COPIED!", bg="#1a472a", fg="#7fcc7f")
        self.root.after(2000, lambda: self.copy_btn.config(
            text="COPY MASTER PROMPT", bg="#0f3460", fg="#a8d8ea"))

    def _on_input_change(self, event=None):
        raw = self.input_text.get("1.0", "end").strip()
        prompts = _extract_image_prompts(raw)
        self.prompts_text.config(state="normal")
        self.prompts_text.delete("1.0", "end")
        self.prompts_text.insert("1.0", prompts)
        self.prompts_text.config(state="disabled")

    def log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ── Generation ───────────────────────────────────────────────────────────
    def _toggle_generation(self):
        raw = self.input_text.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("Empty Input", "Please paste the LLM output.")
            return

        try:
            script_data = _extract_json(raw)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Could not parse JSON:\n{e}")
            return

        if not isinstance(script_data, list) or len(script_data) != 3:
            messagebox.showerror("Invalid Script", "Must contain exactly 3 pairs.")
            return

        # Check images
        missing = [(i, v.get()) for i, v in enumerate(self.image_paths) if not v.get()]
        if missing:
            idx = missing[0][0]
            pair_label = f"Pair {idx//2 + 1}, Image {'A' if idx%2==0 else 'B'}"
            messagebox.showerror("Missing Image", f"Please select an image for {pair_label}.")
            return

        self.cancel_event.clear()
        self.progress_var.set(0)
        self.gen_btn.config(text="CANCEL", bg="#8b0000", fg="white",
                            command=self._request_cancel)
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(script_data,),
            daemon=True
        )
        thread.start()

    def _request_cancel(self):
        self.cancel_event.set()
        self.gen_btn.config(state="disabled", text="CANCELLING...")

    def _reset_button(self):
        self.gen_btn.config(
            text="GENERATE VIDEO", bg="#e94560", fg="white",
            state="normal", command=self._toggle_generation
        )

    def _run_pipeline(self, script_data):
        try:
            img_paths = [v.get() for v in self.image_paths]
            bg_music = self.bg_music_var.get().strip() or None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"comparison_{ts}.mp4"

            self.status_var.set("Rendering...")
            self.log("Starting video generation...")

            def ui_log(msg):
                self.log(msg)

            render_comparison_video(
                script_data=script_data,
                image_paths=img_paths,
                char_dir=CHAR_DIR,
                bg_music_path=bg_music,
                output_path=str(out_path),
                log_fn=ui_log,
                cancel_event=self.cancel_event,
            )

            self.status_var.set(f"Done! Video saved to {out_path.name}")
            self.log(f"Output: {out_path}")
            self.progress_var.set(100)
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.root.after(0, self._reset_button)


if __name__ == "__main__":
    root = tk.Tk()
    app  = ComparisonVideoApp(root)
    root.mainloop()
