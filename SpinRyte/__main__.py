import cv2
import numpy as np
from PIL import ImageGrab
import tkinter as tk
import time

# Paths to template images for each stance (ensure these paths are correct)
template_regular = cv2.imread('templates/regular.png', 0)  # Example: an image of "RE"
template_switch = cv2.imread('templates/switch.png', 0)     # Example: an image of "SW"
template_nollie = cv2.imread('templates/nollie.png', 0)     # Example: an image of "NO"
template_fakie = cv2.imread('templates/fakie.png', 0)       # Example: an image of "FA"

# Convert templates to RGB
template_regular = cv2.cvtColor(template_regular, cv2.COLOR_BGR2RGB)
template_switch = cv2.cvtColor(template_switch, cv2.COLOR_BGR2RGB)
template_nollie = cv2.cvtColor(template_nollie, cv2.COLOR_BGR2RGB)
template_fakie = cv2.cvtColor(template_fakie, cv2.COLOR_BGR2RGB)

# Adjusted coordinates for the stance region (restore original position and reduce height by 300 pixels)
stance_region = (400, 250, 720, 350)  # Region to capture the stance text

class TransparentOverlay:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-transparentcolor", "black")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.5)  # Adjust transparency level (0.0 to 1.0)
        self.root.geometry("2560x1600+0+0")  # Size and position of the window
        self.canvas = tk.Canvas(root, width=2560, height=1600, bg='black', highlightthickness=0)
        self.canvas.pack()

    def draw_arrow(self, direction):
        self.canvas.delete("all")
        arrow_size = 100
        center_y = 1600 // 2  # Keep the Y-position at the vertical center

        if direction == 'left':
            # Position the left arrow on the left side of the screen but centered horizontally
            left_x = 2560 * 0.25  # Quarter of the screen width for left arrow
            center_y = 2560 // 4
            self.canvas.create_polygon(
                [left_x - arrow_size, center_y,
                left_x + arrow_size, center_y - arrow_size,
                left_x + arrow_size, center_y + arrow_size],
                fill="white",
                outline="green"
            )
        elif direction == 'right':
            # Position the right arrow near the right edge of the screen
            right_x = 2560 * 0.4
            center_y = 2560 // 4
            self.canvas.create_polygon(
                [right_x + arrow_size, center_y,
                right_x - arrow_size, center_y - arrow_size,
                right_x - arrow_size, center_y + arrow_size],
                fill="white",
                outline="green"
            )

    def get_stance(self, screenshot):
        """Template matching for stance detection using OpenCV."""
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        
        # Perform template matching for each stance
        res_regular = cv2.matchTemplate(screenshot_rgb, template_regular, cv2.TM_CCOEFF_NORMED)
        res_switch = cv2.matchTemplate(screenshot_rgb, template_switch, cv2.TM_CCOEFF_NORMED)
        res_nollie = cv2.matchTemplate(screenshot_rgb, template_nollie, cv2.TM_CCOEFF_NORMED)
        res_fakie = cv2.matchTemplate(screenshot_rgb, template_fakie, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match for each stance (adjust threshold for accuracy)
        threshold = 0.3
        max_val_regular = np.max(res_regular)
        max_val_switch = np.max(res_switch)
        max_val_nollie = np.max(res_nollie)
        max_val_fakie = np.max(res_fakie)
        
        # Debugging: Print out max values for each stance
        print(f"Regular: {max_val_regular}, Switch: {max_val_switch}, Nollie: {max_val_nollie}, Fakie: {max_val_fakie}")

        # Create a dictionary to map stance to its max value
        stance_scores = {
            'RE': max_val_regular,
            'SW': max_val_switch,
            'NO': max_val_nollie,
            'FA': max_val_fakie
        }

        # Get the stance with the highest score
        best_stance = max(stance_scores, key=stance_scores.get)
        best_score = stance_scores[best_stance]

        # Check if the highest score meets the threshold
        if best_score >= threshold:
            return best_stance
        else:
            return ''  # No stance detected

    def run(self):
        while True:
            # Capture the stance area
            screenshot = np.array(ImageGrab.grab(bbox=stance_region))

            stance = self.get_stance(screenshot)

            if stance == 'RE':  # Regular or no text detected
                self.draw_arrow('right')  # Spin right in regular stance
            elif stance == 'SW':
                self.draw_arrow('left')  # Spin left in switch stance
            elif stance == 'NO':
                self.draw_arrow('right')  # Spin right in nollie stance
            elif stance == 'FA':
                self.draw_arrow('left')  # Spin left in fakie stance
            else:
                # If no stance is detected, default to regular
                print("No stance detected, defaulting to 'regular'.")
                self.draw_arrow('right')

            # Update the Tkinter window and reduce sleep time for quicker updates
            self.root.update()
            time.sleep(0.2)  # Adjust the sleep time as needed for responsiveness

if __name__ == "__main__":
    root = tk.Tk()
    overlay = TransparentOverlay(root)
    overlay.run()
