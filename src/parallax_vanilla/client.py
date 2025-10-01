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
        running = self.running
        window_name = self.window_name
        get_mouse_delta = self.get_mouse_delta
        compute_parallax = self.parallax_model.compute_parallax

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)

        if self.parallax_model.get_color_mode() != "BGR":
            self.parallax_model.convert_color_mode("BGR")


        import time

        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps_display_interval = 1.0  # 每秒更新一次FPS显示
        current_fps = 0.0

        while running:
            # t1 = time.time()
            dx, dy = get_mouse_delta()
            # print(f"mouse delta time: {time.time() - t1}")
            # print(f"Mouse moves: ({dx:.2f}, {dy:.2f})")
            # frame = np.ones([1080, 1920])
            # t1 = time.time()
            frame = compute_parallax(dx, dy)
            # print(f"compute parallax time: {time.time() - t1}")
            if frame is not None:
                cv2.imshow(window_name, frame)
            
            frame_count += 1
            current_time = time.time()
            
            # update FPS once per second
            if current_time - last_fps_update >= fps_display_interval:
                elapsed_time = current_time - start_time
                current_fps = frame_count / elapsed_time
                
                print(f"Mouse: ({dx:.3f}, {dy:.3f}) | Average FPS: {current_fps:.1f} | Total Frames: {frame_count}")
                
                last_fps_update = current_time

            key = cv2.waitKey(1)
            if key == 27 & 0xFF:  # enter ESC to exit 
                running = False
        cv2.destroyAllWindows()

        self.running = running
        print("Parallax display stopped.")

        total_time = time.time() - start_time
        final_avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n=== Parallax Display Statistics ===")
        print(f"Total Runtime: {total_time:.2f} seconds")
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {final_avg_fps:.1f}")
        print("Parallax display stopped.")

if __name__ == "__main__":
    display = ParallaxDisplay(None)
    display.run()