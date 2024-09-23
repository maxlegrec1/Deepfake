import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


class MaskDrawingTool:
    def __init__(self, root, image_dir):
        self.root = root
        self.root.title("Enhanced Segmentation Mask Tool")
        self.image_dir = image_dir
        self.mask_dir = os.path.join(self.image_dir, "masks")
        os.makedirs(self.mask_dir, exist_ok=True)

        self.image_list = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.current_image_index = self.load_progress()
        self.masks = {
            "face": None,
            "mouth": None,
            "eye1": None,
            "eye2": None,
        }
        self.current_mask_type = "face"
        self.image_label = None
        self.image_canvas = None
        self.drawn_shapes = []
        self.brush_size = 5
        self.mask_opacity = 128
        self.display_size = (800, 600)
        self.original_image_size = None
        self.scale_x = 1
        self.scale_y = 1
        self.min_display_size = 400
        self.is_erasing = False

        self.create_widgets()

        if self.image_list:
            self.load_image(self.image_list[self.current_image_index])

    def create_widgets(self):
        # Canvas for the image
        self.image_canvas = tk.Canvas(self.root, width=800, height=600)
        self.image_canvas.pack()

        # Frame for buttons and sliders
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X)

        # Buttons for choosing mask types
        mask_types = ["face", "mouth", "eye1", "eye2"]
        for mask_type in mask_types:
            button = tk.Button(
                control_frame,
                text=f"{mask_type.capitalize()} Mask",
                command=lambda t=mask_type: self.set_mask_type(t),
            )
            button.pack(side=tk.LEFT)

        # Eraser button
        self.eraser_button = tk.Button(
            control_frame, text="Eraser", command=self.toggle_eraser
        )
        self.eraser_button.pack(side=tk.LEFT)

        # Slider for brush size
        self.brush_slider = tk.Scale(
            control_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            label="Brush Size",
            command=self.change_brush_size,
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT)

        # Slider for transparency
        transparency_slider = tk.Scale(
            control_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            label="Mask Transparency",
            command=self.change_opacity,
        )
        transparency_slider.set(self.mask_opacity)
        transparency_slider.pack(side=tk.LEFT)

        # Save and next image button
        next_button = tk.Button(
            control_frame, text="Save and Next", command=self.save_and_next
        )
        next_button.pack(side=tk.RIGHT)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Brush size preview
        self.brush_preview = tk.Canvas(self.root, width=50, height=50)
        self.brush_preview.pack(side=tk.BOTTOM)

        # Bind drawing events to the canvas
        self.image_canvas.bind("<B1-Motion>", self.draw_mask)
        self.image_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Update progress bar
        self.update_progress_bar()

    def load_image(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        self.current_image = Image.open(image_path)
        self.original_image_size = self.current_image.size

        # Resize image if necessary
        if (
            self.current_image.size[0] < self.min_display_size
            or self.current_image.size[1] < self.min_display_size
        ):
            aspect_ratio = self.current_image.size[0] / self.current_image.size[1]
            if aspect_ratio > 1:
                new_size = (
                    self.min_display_size,
                    int(self.min_display_size / aspect_ratio),
                )
            else:
                new_size = (
                    int(self.min_display_size * aspect_ratio),
                    self.min_display_size,
                )
            self.current_image = self.current_image.resize(new_size, Image.LANCZOS)
        elif (
            self.current_image.size[0] > self.display_size[0]
            or self.current_image.size[1] > self.display_size[1]
        ):
            self.current_image.thumbnail(self.display_size, Image.LANCZOS)

        self.scale_x = self.original_image_size[0] / self.current_image.size[0]
        self.scale_y = self.original_image_size[1] / self.current_image.size[1]

        self.tk_image = ImageTk.PhotoImage(self.current_image)
        self.image_canvas.config(
            width=self.current_image.size[0], height=self.current_image.size[1]
        )
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.load_existing_masks()
        self.update_canvas_with_mask()

    def load_existing_masks(self):
        for mask_type in self.masks:
            mask_path = os.path.join(
                self.mask_dir,
                f"{self.image_list[self.current_image_index]}_{mask_type}_mask.png",
            )
            if os.path.exists(mask_path):
                self.masks[mask_type] = Image.open(mask_path).convert("L")
            else:
                self.masks[mask_type] = Image.new("L", self.original_image_size, 0)

    def set_mask_type(self, mask_type):
        self.current_mask_type = mask_type
        self.is_erasing = False
        self.eraser_button.config(relief=tk.RAISED)
        self.update_canvas_with_mask()

    def toggle_eraser(self):
        self.is_erasing = not self.is_erasing
        self.eraser_button.config(relief=tk.SUNKEN if self.is_erasing else tk.RAISED)

    def change_brush_size(self, val):
        self.brush_size = int(val)
        self.update_brush_preview()

    def update_brush_preview(self):
        self.brush_preview.delete("all")
        self.brush_preview.create_oval(
            25 - self.brush_size,
            25 - self.brush_size,
            25 + self.brush_size,
            25 + self.brush_size,
            fill="black" if not self.is_erasing else "white",
            outline="gray",
        )

    def change_opacity(self, val):
        self.mask_opacity = int(val)
        self.update_canvas_with_mask()

    def draw_mask(self, event):
        x, y = int(event.x * self.scale_x), int(event.y * self.scale_y)

        draw = ImageDraw.Draw(self.masks[self.current_mask_type])
        radius = int(self.brush_size * self.scale_x)
        color = 0 if self.is_erasing else 255
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

        canvas_radius = self.brush_size
        self.image_canvas.create_oval(
            event.x - canvas_radius,
            event.y - canvas_radius,
            event.x + canvas_radius,
            event.y + canvas_radius,
            fill="white" if self.is_erasing else "red",
            outline="white" if self.is_erasing else "red",
        )

    def stop_drawing(self, event):
        self.update_canvas_with_mask()

    def update_canvas_with_mask(self):
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        current_mask = self.masks[self.current_mask_type]
        display_mask = current_mask.resize(self.current_image.size, Image.NEAREST)

        red_overlay = Image.new(
            "RGBA", display_mask.size, (255, 0, 0, self.mask_opacity)
        )
        mask_overlay = Image.composite(
            red_overlay,
            Image.new("RGBA", display_mask.size, (0, 0, 0, 0)),
            display_mask,
        )

        merged_image = Image.alpha_composite(
            self.current_image.convert("RGBA"), mask_overlay
        )
        self.tk_mask_image = ImageTk.PhotoImage(merged_image)

        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_mask_image)

    def save_and_next(self):
        self.save_masks()
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_list):
            messagebox.showinfo("Info", "All images processed!")
            self.save_progress()
            self.root.quit()
        else:
            self.save_progress()
            self.load_image(self.image_list[self.current_image_index])
        self.update_progress_bar()

    def save_masks(self):
        for mask_type, mask_image in self.masks.items():
            mask_path = os.path.join(
                self.mask_dir,
                f"{self.image_list[self.current_image_index]}_{mask_type}_mask.png",
            )
            mask_image.save(mask_path)

    def save_progress(self):
        progress_file = os.path.join(self.mask_dir, "progress.pkl")
        with open(progress_file, "wb") as f:
            pickle.dump(self.current_image_index, f)

    def load_progress(self):
        progress_file = os.path.join(self.mask_dir, "progress.pkl")
        if os.path.exists(progress_file):
            with open(progress_file, "rb") as f:
                return pickle.load(f)
        return 0

    def update_progress_bar(self):
        progress = (self.current_image_index / len(self.image_list)) * 100
        self.progress_var.set(progress)


if __name__ == "__main__":
    root = tk.Tk()
    image_dir = filedialog.askdirectory(title="Select Image Directory")
    if image_dir:
        app = MaskDrawingTool(root, image_dir)
        root.mainloop()
