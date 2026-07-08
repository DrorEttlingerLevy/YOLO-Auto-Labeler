import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
from functools import partial

images_dir = r"G:\My Drive\PhD\video YOLO\final project ML course\extra images"
labels_dir = r"G:\My Drive\PhD\video YOLO\final project ML course\data\output_folder_auto_labeler"
model_path = r"G:\My Drive\PhD\video YOLO\final project ML course\data\runs\detect\yolov8-transfer-fish4\weights\best.pt"

os.makedirs(labels_dir, exist_ok=True)
model = YOLO(model_path)

# All images in the folder
all_images = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
# All label files in the folder
all_labels = [
    f for f in os.listdir(labels_dir)
    if f.lower().endswith('.txt')
]
# Only work on images that do not already have a label file
images = [
    f for f in all_images
    if not os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt'))
]

total_images = len(all_images)
labeled_count = len(all_labels)

def get_current_labeled_count():
    """Get the current count of labeled images"""
    return len([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')])

class LabelReviewer:
    def __init__(self, master):
        self.master = master
        self.index = 0
        self.box_states = []  # True=accept, False=reject
        self.box_ids = []
        self.box_coords = []  # Store display coordinates for hit-testing and new boxes
        self.manual_boxes = []  # List of manually added boxes (in display coordinates)
        self.manual_box_states = []  # States for manual boxes
        self.manual_box_ids = []  # Canvas IDs for manual boxes
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.canvas.pack()
        self.image_label = tk.Label(self.main_frame, text="", font=("Arial", 14, "bold"))
        self.image_label.pack()
        self.progress_label = tk.Label(self.main_frame, text="", font=("Arial", 12))
        self.progress_label.pack()
        self.add_box_btn = tk.Button(self.main_frame, text="Add Box", command=self.start_add_box)
        self.add_box_btn.pack(side=tk.LEFT)
        # Right panel for BB crops (no scroll, dynamic size)
        self.right_panel = tk.Frame(master)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thumbs_frame = tk.Frame(self.right_panel)
        self.thumbs_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.bb_thumbnails = []  # Store references to thumbnail widgets
        self.bb_images = []      # Store references to PhotoImage objects
        self.save_next_btn = tk.Button(self.right_panel, text="Save & Next", command=self.save_and_next)
        self.save_next_btn.pack(side=tk.TOP, fill=tk.X, pady=10)
        # Hotkey explanation label
        self.hotkey_label = tk.Label(self.right_panel, text="Hotkeys: n = Add Box, m = Save & Next", font=("Arial", 10))
        self.hotkey_label.pack(side=tk.BOTTOM, pady=10)
        self.img_id = None
        self.img = None
        self.tk_img = None
        self.detections = None
        self.adding_box = False
        self.start_x = self.start_y = None
        self.temp_rect = None
        self.show_image()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonPress-3>", self.start_add_box)  # Right click to add box (optional)
        self.master.bind("n", self.on_n_key)
        self.master.bind("N", self.on_n_key)
        self.master.bind("m", self.on_m_key)
        self.master.bind("M", self.on_m_key)
        self.highlighted_bb = None  # Index of currently highlighted BB

    def on_n_key(self, event):
        self.start_add_box()

    def on_m_key(self, event):
        self.save_and_next()

    def draw_boxes(self):
        self.canvas.delete("all")
        if self.tk_img is not None:
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.box_ids = []
        self.box_coords = []
        if not self.detections or not hasattr(self.detections, 'boxes') or self.detections.boxes is None:
            return
        img_h, img_w = self.img.shape[:2]
        disp_w, disp_h = 640, 480
        scale_x = disp_w / img_w
        scale_y = disp_h / img_h
        for i, box in enumerate(self.detections.boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            x1_disp = int(x1 * scale_x)
            y1_disp = int(y1 * scale_y)
            x2_disp = int(x2 * scale_x)
            y2_disp = int(y2 * scale_y)
            if self.highlighted_bb == i:
                color = "yellow"
                width = 4
            else:
                color = "red" if self.box_states[i] else "gray"
                width = 2
            rect_id = self.canvas.create_rectangle(x1_disp, y1_disp, x2_disp, y2_disp, outline=color, width=width)
            text_id = self.canvas.create_text(x1_disp + 4, y1_disp + 12, anchor="nw", text=str(i), fill="white", font=("Arial", 10))
            self.box_ids.append((rect_id, text_id))
            self.box_coords.append((x1_disp, y1_disp, x2_disp, y2_disp))
        # Draw manual boxes
        for i, (x1, y1, x2, y2) in enumerate(self.manual_boxes):
            color = "red" if self.manual_box_states[i] else "gray"
            rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, dash=(2,2))
            text_id = self.canvas.create_text(x1 + 4, y1 + 12, anchor="nw", text=f"M{i}", fill="yellow", font=("Arial", 10))
            self.manual_box_ids.append((rect_id, text_id))
        self.draw_bb_thumbnails()

    def draw_bb_thumbnails(self):
        # Clear previous thumbnails
        for widget in self.bb_thumbnails:
            widget.destroy()
        self.bb_thumbnails = []
        self.bb_images = []
        for widget in self.thumbs_frame.winfo_children():
            widget.destroy()
        if self.img is None or not self.detections or not hasattr(self.detections, 'boxes') or self.detections.boxes is None:
            return
        img_h, img_w = self.img.shape[:2]
        max_cols = 5
        n_bbs = len(self.detections.boxes.xyxy)
        n_rows = (n_bbs + max_cols - 1) // max_cols
        thumb_size = 80
        self.right_panel.update_idletasks()
        self.right_panel.config(width=max_cols*thumb_size+40, height=n_rows*thumb_size+100)
        self.thumbs_frame.config(width=max_cols*thumb_size, height=n_rows*thumb_size)
        # Sort indices by confidence (highest to lowest)
        if hasattr(self.detections.boxes, 'conf') and self.detections.boxes.conf is not None:
            confs = self.detections.boxes.conf.tolist()
            sorted_indices = sorted(range(n_bbs), key=lambda i: confs[i], reverse=True)
        else:
            sorted_indices = list(range(n_bbs))
        col = 0
        row = 0
        for grid_idx, i in enumerate(sorted_indices):
            box = self.detections.boxes.xyxy[i]
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = self.img[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop).resize((thumb_size, thumb_size))
            crop_imgtk = ImageTk.PhotoImage(crop_pil)
            self.bb_images.append(crop_imgtk)
            frame = tk.Frame(self.thumbs_frame, bd=2, relief=tk.RIDGE)
            # Add tag number and confidence above the thumbnail
            tag_text = f"{i}"
            if hasattr(self.detections.boxes, 'conf') and self.detections.boxes.conf is not None:
                conf_val = self.detections.boxes.conf[i]
                tag_text += f" ({conf_val:.2f})"
            tag_label = tk.Label(frame, text=tag_text, font=("Arial", 10, "bold"))
            tag_label.pack()
            label = tk.Label(frame, image=crop_imgtk)
            label.pack()
            btn_text = "Reject" if self.box_states[i] else "Accept"
            btn_color = "red" if self.box_states[i] else "gray"
            btn = tk.Button(frame, text=btn_text, bg=btn_color, fg="white", command=lambda idx=i: self.toggle_bb_from_thumb(idx))
            btn.pack(fill=tk.X)
            frame.grid(row=row, column=col, padx=2, pady=2, sticky="n")
            # Highlight on click only (do not toggle accept/reject)
            def on_click(event, idx=i):
                if self.highlighted_bb == idx:
                    self.highlighted_bb = None
                else:
                    self.highlighted_bb = idx
                self.draw_boxes()
            label.bind("<Button-1>", on_click)
            frame.bind("<Button-1>", on_click)
            self.bb_thumbnails.append(frame)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        # Add/update legend for confidence
        if not hasattr(self, 'legend_label'):
            self.legend_label = tk.Label(self.right_panel, text="( ) = confidence level", font=("Arial", 9, "italic"))
            self.legend_label.pack(side=tk.BOTTOM, pady=2)
        else:
            self.legend_label.config(text="( ) = confidence level")

    def toggle_bb_from_thumb(self, idx):
        self.toggle_box_state(idx)
        self.draw_bb_thumbnails()
        self.draw_boxes()

    def on_canvas_click(self, event):
        if self.adding_box:
            return  # Ignore clicks while adding a box
        # Check if click is on any box edge/label (by canvas id)
        clicked_item = self.canvas.find_withtag('current')
        for idx, (rect_id, text_id) in enumerate(self.box_ids):
            if clicked_item and (clicked_item[0] == rect_id or clicked_item[0] == text_id):
                self.toggle_box_state(idx)
                return
        for idx, (rect_id, text_id) in enumerate(self.manual_box_ids):
            if clicked_item and (clicked_item[0] == rect_id or clicked_item[0] == text_id):
                self.toggle_manual_box_state(idx)
                return
        # If not on edge/label, check if click is inside any box
        auto_hits = [i for i, (x1, y1, x2, y2) in enumerate(self.box_coords) if x1 <= event.x <= x2 and y1 <= event.y <= y2]
        manual_hits = [i for i, (x1, y1, x2, y2) in enumerate(self.manual_boxes) if x1 <= event.x <= x2 and y1 <= event.y <= y2]
        if manual_hits:
            self.toggle_manual_box_state(manual_hits[-1])
            return
        if auto_hits:
            self.toggle_box_state(auto_hits[-1])
            return

    def toggle_box_state(self, idx):
        self.box_states[idx] = not self.box_states[idx]
        color = "red" if self.box_states[idx] else "gray"
        self.canvas.itemconfig(self.box_ids[idx][0], outline=color)

    def toggle_manual_box_state(self, idx):
        self.manual_box_states[idx] = not self.manual_box_states[idx]
        color = "red" if self.manual_box_states[idx] else "gray"
        self.canvas.itemconfig(self.manual_box_ids[idx][0], outline=color)

    def start_add_box(self, event=None):
        self.adding_box = True
        self.canvas.bind("<Button-1>", self.add_box_press)
        self.canvas.bind("<B1-Motion>", self.add_box_drag)
        self.canvas.bind("<ButtonRelease-1>", self.add_box_release)

    def add_box_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
        self.temp_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="yellow", dash=(2,2))

    def add_box_drag(self, event):
        if self.temp_rect:
            self.canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def add_box_release(self, event):
        end_x, end_y = event.x, event.y
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        self.manual_boxes.append((x1, y1, x2, y2))
        self.manual_box_states.append(True)
        self.manual_box_ids = []  # Will be redrawn
        self.adding_box = False
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.draw_boxes()

    def save_and_next(self):
        name = images[self.index]
        label_path = os.path.join(labels_dir, os.path.splitext(name)[0] + '.txt')
        if self.img is None or self.detections is None or not hasattr(self.detections, 'boxes') or self.detections.boxes is None:
            # No detections or image, just move on
            self.index += 1
            if self.index < len(images):
                self.show_image()
            else:
                self.master.destroy()
            return
        img_h, img_w = self.img.shape[:2]
        disp_w, disp_h = 640, 480
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h
        with open(label_path, 'w') as f:
            # Save accepted auto-detected boxes
            for i, box in enumerate(self.detections.boxes.xyxy):
                if not self.box_states[i]:
                    continue
                x1, y1, x2, y2 = box.tolist()
                if hasattr(self.detections.boxes, 'cls'):
                    cls = int(self.detections.boxes.cls[i])
                else:
                    cls = 0
                x_c = (x1 + x2) / 2 / img_w
                y_c = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            # Save accepted manual boxes (as class 0)
            for i, (x1_disp, y1_disp, x2_disp, y2_disp) in enumerate(self.manual_boxes):
                if not self.manual_box_states[i]:
                    continue
                # Convert display coords to original image coords
                x1 = x1_disp * scale_x
                y1 = y1_disp * scale_y
                x2 = x2_disp * scale_x
                y2 = y2_disp * scale_y
                x_c = (x1 + x2) / 2 / img_w
                y_c = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        # Move to next image after saving label
        self.index += 1
        if self.index < len(images):
            self.show_image()
        else:
            self.master.destroy()

    def show_image(self):
        name = images[self.index]
        path = os.path.join(images_dir, name)
        self.img = cv2.imread(path)
        self.detections = model(path)[0]
        n_boxes = 0
        if self.detections and hasattr(self.detections, 'boxes') and self.detections.boxes is not None:
            n_boxes = len(self.detections.boxes.xyxy)
        self.box_states = [True] * n_boxes  # All boxes accepted by default
        self.manual_boxes = []
        self.manual_box_states = []
        self.manual_box_ids = []
        self.highlighted_bb = None # Reset highlighted box
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(img_pil.resize((640, 480)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.image_label.config(text=name)
        # Show progress: labeled out of all images in the folder
        current_labeled = get_current_labeled_count()
        self.progress_label.config(text=f"Labeled: {current_labeled} / {total_images} images")
        self.draw_boxes()

root = tk.Tk()
root.title("Label Reviewer")
app = LabelReviewer(root)
root.mainloop()
