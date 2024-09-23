import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk


class MaskDrawingTool:
    def __init__(self, root, image_dir):
        self.root = root
        self.root.title("Segmentation Mask Tool")
        self.image_dir = image_dir
        self.mask_dir = os.path.join(self.image_dir, "masks")  # Separate mask directory
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
        }  # Separate masks for each type
        self.current_mask_type = "face"  # Default mask type
        self.image_label = None
        self.image_canvas = None
        self.drawn_shapes = []  # To store all drawn shapes
        self.brush_size = 5  # Default brush size
        self.mask_opacity = 128  # Default opacity for mask (0-255)
        self.display_size = (800, 600)  # Size for displaying images
        self.original_image_size = None  # To keep track of the original image size

        # Create Buttons, Canvas, and Brush Size Slider
        self.create_widgets()

        # Load the first image
        if self.image_list:
            self.load_image(self.image_list[self.current_image_index])

    def create_widgets(self):
        # Canvas for the image
        self.image_canvas = tk.Canvas(self.root, width=800, height=600)
        self.image_canvas.pack()

        # Buttons for choosing mask types
        face_button = tk.Button(
            self.root, text="Face Mask", command=lambda: self.set_mask_type("face")
        )
        face_button.pack(side=tk.LEFT)

        mouth_button = tk.Button(
            self.root, text="Mouth Mask", command=lambda: self.set_mask_type("mouth")
        )
        mouth_button.pack(side=tk.LEFT)

        eye1_button = tk.Button(
            self.root, text="Left Eye Mask", command=lambda: self.set_mask_type("eye1")
        )
        eye1_button.pack(side=tk.LEFT)

        eye2_button = tk.Button(
            self.root, text="Right Eye Mask", command=lambda: self.set_mask_type("eye2")
        )
        eye2_button.pack(side=tk.LEFT)

        # Slider for brush size
        brush_slider = tk.Scale(
            self.root,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            label="Brush Size",
            command=self.change_brush_size,
        )
        brush_slider.set(self.brush_size)  # Set initial brush size
        brush_slider.pack(side=tk.LEFT)

        # Slider for transparency
        transparency_slider = tk.Scale(
            self.root,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            label="Mask Transparency",
            command=self.change_opacity,
        )
        transparency_slider.set(self.mask_opacity)  # Set initial opacity
        transparency_slider.pack(side=tk.LEFT)

        # Save and next image button
        next_button = tk.Button(
            self.root, text="Save and Next", command=self.save_and_next
        )
        next_button.pack(side=tk.RIGHT)

        # Bind drawing events to the canvas
        self.image_canvas.bind("<B1-Motion>", self.draw_mask)
        self.image_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def load_image(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        self.current_image = Image.open(image_path)
        self.original_image_size = self.current_image.size  # Store original size

        # Check if image needs to be resized for better visibility
        if (
            self.current_image.size[0] < self.display_size[0]
            or self.current_image.size[1] < self.display_size[1]
        ):
            # Resize image for display only (keeping aspect ratio)
            self.current_image.thumbnail(
                self.display_size
            )  # Resize to fit within 800x600 for better viewing
            print(
                f"Resized image from {self.original_image_size} to {self.current_image.size} for display"
            )

        self.tk_image = ImageTk.PhotoImage(self.current_image)
        self.image_canvas.delete("all")  # Clear canvas before loading new image
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Create blank masks at original size (for saving correctly)
        self.masks = {
            "face": Image.new("L", self.original_image_size, 0),
            "mouth": Image.new("L", self.original_image_size, 0),
            "eye1": Image.new("L", self.original_image_size, 0),
            "eye2": Image.new("L", self.original_image_size, 0),
        }

    def set_mask_type(self, mask_type):
        self.current_mask_type = mask_type
        print(f"Mask type set to: {mask_type}")
        self.update_canvas_with_mask()

    def change_brush_size(self, val):
        self.brush_size = int(val)  # Update brush size
        print(f"Brush size set to: {self.brush_size}")

    def change_opacity(self, val):
        self.mask_opacity = int(val)  # Update transparency
        self.update_canvas_with_mask()

    def draw_mask(self, event):
        # Convert canvas coordinates to original image coordinates if resized
        scale_x = self.original_image_size[0] / self.current_image.size[0]
        scale_y = self.original_image_size[1] / self.current_image.size[1]
        x, y = int(event.x * scale_x), int(event.y * scale_y)

        draw = ImageDraw.Draw(self.masks[self.current_mask_type])
        radius = self.brush_size  # Use the updated brush size
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

        # Update the canvas with a small dot
        self.image_canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="red",
            outline="red",
        )

    def stop_drawing(self, event):
        self.update_canvas_with_mask()  # Refresh the mask display

    def update_canvas_with_mask(self):
        """Clear canvas and display the current image and only the selected mask with adjustable transparency."""
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Only display the current mask with the selected transparency
        current_mask = self.masks[self.current_mask_type]

        # Resize mask for display if image was resized
        if self.current_image.size != self.original_image_size:
            display_mask = current_mask.resize(self.current_image.size, Image.NEAREST)
        else:
            display_mask = current_mask

        # Convert the current mask into a red overlay with the selected opacity
        red_overlay = Image.new(
            "RGBA", display_mask.size, (255, 0, 0, self.mask_opacity)
        )
        mask_overlay = Image.composite(
            red_overlay,
            Image.new("RGBA", display_mask.size, (0, 0, 0, 0)),
            display_mask,
        )

        # Merge the mask overlay with the original image
        merged_image = Image.alpha_composite(
            self.current_image.convert("RGBA"), mask_overlay
        )
        self.tk_mask_image = ImageTk.PhotoImage(merged_image)

        # Display the updated image with the selected mask
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_mask_image)

    def save_and_next(self):
        # Save the current mask
        self.save_masks()

        # Move to the next image
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_list):
            messagebox.showinfo("Info", "All images processed!")
            self.save_progress()  # Save final progress
            self.root.quit()
        else:
            self.save_progress()
            self.load_image(self.image_list[self.current_image_index])

    def save_masks(self):
        # Save all masks as separate image files for the current image in the masks directory
        for mask_type, mask_image in self.masks.items():
            mask_path = os.path.join(
                self.mask_dir,
                f"{self.image_list[self.current_image_index]}_{mask_type}_mask.png",
            )
            mask_image.save(mask_path)
            print(
                f"{mask_type.capitalize()} mask saved for {self.image_list[self.current_image_index]}"
            )

    def save_progress(self):
        # Save the current image index to resume later
        progress_file = os.path.join(self.mask_dir, "progress.pkl")
        with open(progress_file, "wb") as f:
            pickle.dump(self.current_image_index, f)

    def load_progress(self):
        # Load progress if exists, otherwise start at 0
        progress_file = os.path.join(self.mask_dir, "progress.pkl")
        if os.path.exists(progress_file):
            with open(progress_file, "rb") as f:
                return pickle.load(f)
        return 0


if __name__ == "__main__":
    root = tk.Tk()
    image_dir = filedialog.askdirectory(title="Select Image Directory")
    if image_dir:
        app = MaskDrawingTool(root, image_dir)
        root.mainloop()
    else:
        print("No directory selected")
