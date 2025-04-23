import asyncio
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkFont
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import aiofiles
import cv2
import numpy as np
import pymupdf
import pytesseract
from aiocsv import AsyncDictWriter
from PIL import Image

# Enable Windows DPI awareness for sharper UI on high-DPI displays
if sys.platform == "win32":
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

# --- Configuration ---
HARDCODED_COORDS = (1030, 2133, 2251, 2685)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
FIELDNAMES = ("name", "address", "city", "area", "phone")


# text parsing from pdf logic
def ocr_region_blocking(
    pdf_path, page_index, coords, dpi=500, lang="eng", oem=3, psm=6
):
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    x0, y0, x1, y1 = coords
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(
        Image.fromarray(thresh), lang=lang, config=f"--oem {oem} --psm {psm}"
    )


def parse_recipient_blocking(text):
    lines = list(map(str.strip, filter(str, text.split("\n"))))
    data = dict.fromkeys(FIELDNAMES, "")
    if len(lines) >= 5:
        data["name"] = lines[0].split("Recipient ")[1]
        data["address"] = " ".join(lines[1:-3])
        data["city"] = lines[-3]
        data["area"] = lines[-2]
        data["phone"] = lines[-1].split(" ")[1]
    return data


# async pipline
async def async_main(pdf_paths, coords, queue):
    page_idx = 0
    dpi = 500
    total = len(pdf_paths)
    records = []

    for idx, pdf in enumerate(pdf_paths):
        ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        queue.put(
            ("log", f"{ts_start} ▶ Starting OCR: {Path(pdf).name} ({idx+1}/{total})")
        )

        # Perform OCR
        text = await asyncio.to_thread(ocr_region_blocking, pdf, page_idx, coords, dpi)
        ts_ocr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        queue.put(("log", f"{ts_ocr} ✔ OCR done: {Path(pdf).name}"))

        # Parse result
        record = await asyncio.to_thread(parse_recipient_blocking, text)
        ts_parsed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parsed_str = "; ".join(f"{k}: {v}" for k, v in record.items())
        queue.put(("log", f"{ts_parsed} → Parsed: {Path(pdf).name} [ {parsed_str} ]"))

        # Update progress bar
        queue.put(("progress", 1))
        records.append(record)

    return records


async def export_to_csv_async(records, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    async with aiofiles.open(save_path, mode="a+", newline="") as afp:
        writer = AsyncDictWriter(afp, fieldnames=FIELDNAMES)
        await afp.seek(0)
        if not await afp.read(1):
            await writer.writeheader()
        await writer.writerows(records)


class PDFProcessorGUI:
    def __init__(self, root):
        self.root = root
        root.title("PDF OCR Processor")
        root.geometry("1200x800")
        root.minsize(1000, 700)
        root.tk.call("tk", "scaling", 2.0)

        self.title_font = tkFont.Font(family="Segoe UI", size=14, weight="bold")
        self.label_font = tkFont.Font(family="Segoe UI", size=12)
        self.log_font = tkFont.Font(family="Consolas", size=10)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=self.label_font, padding=6)
        style.configure("TLabel", font=self.label_font)
        style.configure("TProgressbar", thickness=20)

        self.pdf_paths = []
        self.save_path = ""
        self.queue = queue.Queue()
        self.worker = None

        # File selection
        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="x")
        ttk.Button(frame, text="Select PDF Files", command=self.select_files).pack(
            side="left"
        )
        self.file_label = ttk.Label(
            frame, text="No files selected", font=self.title_font
        )
        self.file_label.pack(side="left", padx=10)

        # Output selection
        frame2 = ttk.Frame(root, padding=10)
        frame2.pack(fill="x")
        ttk.Button(frame2, text="Choose Output CSV", command=self.choose_output).pack(
            side="left"
        )
        self.out_label = ttk.Label(
            frame2, text="No output chosen", font=self.title_font
        )
        self.out_label.pack(side="left", padx=10)

        # Progress bar and percent
        self.progress = ttk.Progressbar(
            root, orient="horizontal", length=800, mode="determinate"
        )
        self.progress.pack(padx=10, pady=(20, 5))
        self.percent_label = ttk.Label(root, text="0%", font=self.title_font)
        self.percent_label.pack(pady=(0, 10))

        # Log pane
        self.log_text = tk.Text(
            root,
            height=12,
            state="disabled",
            font=self.log_font,
            bg="#1e1e1e",
            fg="#d4d4d4",
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

        # Start button and status
        self.start_btn = ttk.Button(
            root,
            text="Start Processing",
            command=self.start_processing,
            state="disabled",
        )
        self.start_btn.pack(fill="x", padx=10, pady=10)
        self.status_label = ttk.Label(root, text="Idle", font=self.title_font)
        self.status_label.pack(padx=10, pady=5)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select PDF files", filetypes=[("PDF files", "*.pdf")]
        )
        if files:
            self.pdf_paths = list(files)
            self.file_label.config(text=f"{len(files)} files selected")
            self.progress["maximum"] = len(files)
            self.enable_start()

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Save CSV As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            confirmoverwrite=False,
        )
        if path:
            self.save_path = path
            self.out_label.config(text=Path(path).name)
            self.enable_start()

    def enable_start(self):
        if self.pdf_paths and self.save_path:
            self.start_btn.config(state="normal")

    def start_processing(self):
        self.start_btn.config(state="disabled")
        self.status_label.config(text="Processing OCR...", foreground="blue")
        # reset UI
        self.progress["value"] = 0
        self.percent_label.config(text="0%")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")
        # background thread
        self.worker = threading.Thread(target=self.run_pipeline, daemon=True)
        self.worker.start()
        self.root.after(100, self.check_queue)

    def run_pipeline(self):
        records = asyncio.run(async_main(self.pdf_paths, HARDCODED_COORDS, self.queue))
        asyncio.run(export_to_csv_async(records, self.save_path))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.queue.put(
            ("log", f"{ts} - All tasks complete. CSV saved to {self.save_path}")
        )
        self.queue.put(("done", None))

    def check_queue(self):
        try:
            while True:
                typ, val = self.queue.get_nowait()
                if typ == "progress":
                    self.progress["value"] += val
                    pct = int((self.progress["value"] / self.progress["maximum"]) * 100)
                    self.percent_label.config(text=f"{pct}%")
                elif typ == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, val + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
                elif typ == "done":
                    self.status_label.config(
                        text="Finished Processing", foreground="green"
                    )
                    messagebox.showinfo("Done", "All PDFs processed and CSV saved.")
        except queue.Empty:
            pass

        if self.worker and self.worker.is_alive():
            self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    PDFProcessorGUI(root)
    root.mainloop()
