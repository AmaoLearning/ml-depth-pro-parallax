import numpy as np
import cv2
from win32 import win32api, win32gui, win32print
from win32.lib import win32con

class ParallaxDisplay:
    def __init__(self, parallx_model, window_name="Parallax Display"):

        self.parallax_model = parallx_model
        self.window_name = window_name
        
        hDesktop = win32gui.GetDesktopWindow()

        hDC = win32gui.GetDC(hDesktop)

        screen_width = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
        screen_height = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)

        win32gui.ReleaseDC(hDesktop, hDC)

        print(f"Load screen size: ({screen_height}, {screen_width})")
        self.center = (screen_width / 2.0, screen_height / 2.0)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.running = False

    def get_mouse_delta(self):
        curr_pos = win32api.GetCursorPos()
        dx = curr_pos[0] - self.center[0]
        dy = curr_pos[1] - self.center[1]
        
        n_dx = dx / self.screen_width
        n_dy = dy / self.screen_height

        n_dx = max(min(0.5, n_dx), -0.5)
        n_dy = max(min(0.5, n_dy), -0.5)
        
        return n_dx, n_dy

    def run(self):
        self.running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if self.parallax_model.get_color_mode() != "BGR":
            self.parallax_model.convert_color_mode("BGR")

        while self.running:
            dx, dy = self.get_mouse_delta()
            print(f"Mouse moves: ({dx:.2f}, {dy:.2f})")
            # frame = np.ones([1080, 1920])
            frame = self.parallax_model.compute_parallax(dx, dy)
            if frame is not None:
                cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1)
            if key == 27:  # enter ESC to exit 
                self.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    display = ParallaxDisplay(None)
    display.run()