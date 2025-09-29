class Mouse:
    def __init__(self, image_size):
        """
        Fake mouse for test

        - `get_move_distance` get your input from keyboard in the form of "x y",\n
            and return the distance each in range [-0.5, 0.5].
        """
        self.bounds = image_size
        self.center = (image_size[0]/2.0, image_size[1]/2.0)
    
    def get_move_distance(self):
        cur = input("Input pixel distance from keyboard in the form of 'x y': ")
        cur = cur.split(' ')
        dx = cur[0]
        dy = cur[1]

        try:
            dx = float(dx)
            dy = float(dy)
        except ValueError:
            print("\nInvalid input. `get_move_distance` get your input from keyboard in the form of 'x y'.")
            return None
        
        n_dx = dx / self.bounds[1]
        n_dy = dy / self.bounds[0]

        n_dx = max(min(0.5, n_dx), -0.5)
        n_dy = max(min(0.5, n_dy), -0.5)

        return n_dx, n_dy

if __name__ == "__main__":
    m = Mouse((1080, 1920))
    print(f"Mouse: center {m.center}, bounds {m.bounds}")

    dx, dy = m.get_move_distance()
    print(f"Mouse: distance {dx}, {dy}")