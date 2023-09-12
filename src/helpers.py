import time


class ChartData(object):
    def __init__(self, size_max=0):
        self.x = []
        self.y = []
        self.last_x = 0
        self.size_max = size_max

    def add_point(self, x, y):
        if (x - self.last_x) ** 2 < 0.0001:
            return
        self.last_x = x
        self.x.append(1.0 * x)
        self.y.append(1.0 * y)
        if 0 < self.size_max < len(self.x):
            self.x.pop(0)
            self.y.pop(0)

    @staticmethod
    def get_scaled_vector(vector, data_range, canvas_width):
        scaled = []
        for val in vector:
            scaled.append(
                (val - data_range[0]) * canvas_width / (data_range[1] - data_range[0])
            )
        return scaled

    def get_line(self, horizontal_range, canvas_width, vertical_range, canvas_height):
        x_scaled = self.get_scaled_vector(self.x, horizontal_range, canvas_width)
        y_scaled = self.get_scaled_vector(self.y, vertical_range, canvas_height)
        line = []
        for i, j in zip(x_scaled, y_scaled):
            if 0 <= i < canvas_width and 0 <= j < canvas_height:
                line.append(i)
                line.append(canvas_height - 1 - j)
        return line


class Chart(object):
    def __init__(self, canvas, horizontal_range, vertical_range, nb_max_points=5000):
        self.canvas = canvas
        self.data = ChartData(nb_max_points)
        self.x_range = horizontal_range
        self.y_range = vertical_range
        self.line = None

    def ensure_visible(self, x, y):
        width = 1.0 * self.canvas.winfo_width()
        height = 1.0 * self.canvas.winfo_height()
        scaled_width = 1.0 * (self.x_range[1] - self.x_range[0])
        scaled_height = 1.0 * (self.y_range[1] - self.y_range[0])
        x_step = scaled_width / width
        y_step = scaled_height / height
        x_max_visible = self.x_range[0] + (width - 1.0) * x_step
        y_max_visible = self.y_range[0] + (height - 1.0) * y_step
        if x > x_max_visible:
            self.x_range = (
                self.x_range[0] + x - x_max_visible,
                self.x_range[1] + x - x_max_visible,
            )
        if x < self.x_range[0]:
            self.x_range = (x, x + self.x_range[1] - self.x_range[0])
        if y >= y_max_visible:
            self.y_range = (self.y_range[0], y + 0.1 * abs(y) + 1e-6)
        if y <= self.y_range[0]:
            self.y_range = (y - 0.1 * abs(y) - 1e-6, self.y_range[1])

    def add_point(self, x, y):
        self.ensure_visible(x, y)
        self.data.add_point(x, y)

    def update(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        l = self.data.get_line(self.x_range, width, self.y_range, height)
        if len(l) > 2:
            new_line = self.canvas.create_line(l)
            if self.line:
                self.canvas.delete(self.line)
            self.line = new_line

    def clean(self):
        self.data.x = []
        self.data.y = []
        self.data.last_x = 0


class Chrono(object):
    def __init__(self):
        self.time_start = time.time()

    def start(self):
        self.time_start = time.time()

    def restart(self):
        ret = time.time() - self.time_start
        self.time_start = time.time()
        return ret

    def get_time(self):
        return time.time() - self.time_start

    def set_time(self, time_origin):
        self.time_start = time.time() - time_origin
