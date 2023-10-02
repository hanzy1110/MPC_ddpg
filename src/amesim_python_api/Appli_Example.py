
import tkinter
import time
import AmeCommunication
import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
# State-space model
class MPCControllerup:
    def __init__(self):
        self.A = np.array([
            [-0.3324, -1.0684, 0, 0, 0, 0],
            [1.0000, 0, 0, 0, 0, 0],
            [0, 0, -0.0474, -0.1713, 0, 0],
            [0, 0, 0.2500, 0, 0, 0],
            [0, 0, 0, 0, -0.0014, -0.1980],
            [0, 0, 0, 0, 0.2500, 0]
        ])

        self.B = np.array([
            [0.0625, 0, 0],
            [0, 0, 0],
            [0, 0.5000, 0],
            [0, 0, 0],
            [0, 0, 0.0312],
            [0, 0, 0]
        ])

        self.C = np.array([
            [-0.0401, -0.0308, -0.3769, 0.2964, 0.0192, 0.0181]
        ])

        self.D = np.array([
            [0, 0, 0]
        ])


        self.dt = 0.1       
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.Ad = sp.linalg.expm(self.A*self.dt)
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad-np.eye(self.nx)) @ self.B
        self.Cd = self.C
        self.Dd = self.D
        self.N = 1      
        self.M = 1       
        self.lam = 1e-5  
    
    def controlup(self, x0, ref):
        
        if isinstance(ref, (float, int)):
            ref = np.ones(self.N) * ref
        elif isinstance(ref, np.ndarray) and ref.shape == ():
            ref = np.ones(self.N) * ref.item()

        
        x = cp.Variable((self.nx, self.N+1))
        u = cp.Variable((self.nu, self.M))
        
        
        objective = 0
        constraints = [x[:, 0] == x0]


        for k in range(self.N):
            objective += (self.Cd @ x[:, k+1] - ref[k]) ** 2
            if k < self.M:
                objective += self.lam * cp.quad_form(u[:, k], np.eye(self.nu))
                constraints += [x[:, k+1] == self.Ad @ x[:, k] + self.Bd @ u[:, k]]
                constraints += [u[:, k] >= -10, u[:, k] <= 0]
            else:
                constraints += [x[:, k+1] == self.Ad @ x[:, k]]
            
        
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        return u[:, 0].value

class MPCControllerdown:
    def __init__(self):
        self.A = np.array([
            [-0.3324, -1.0684, 0, 0, 0, 0],
            [1.0000, 0, 0, 0, 0, 0],
            [0, 0, -0.0474, -0.1713, 0, 0],
            [0, 0, 0.2500, 0, 0, 0],
            [0, 0, 0, 0, -0.0014, -0.1980],
            [0, 0, 0, 0, 0.2500, 0]
        ])

        self.B = np.array([
            [0.0625, 0, 0],
            [0, 0, 0],
            [0, 0.5000, 0],
            [0, 0, 0],
            [0, 0, 0.0312],
            [0, 0, 0]
        ])

        self.C = np.array([
            [-0.0401, -0.0308, -0.3769, 0.2964, 0.0192, 0.0181]
        ])

        self.D = np.array([
            [0, 0, 0]
        ])

        self.dt = 0.1      
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.Ad = sp.linalg.expm(self.A*self.dt)
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad-np.eye(self.nx)) @ self.B
        self.Cd = self.C
        self.Dd = self.D
        self.N = 1      
        self.M = 1       
        self.lam = 1e-5  
    
    def controldown(self, x0, ref):
        
        if isinstance(ref, (float, int)):
            ref = np.ones(self.N) * ref
        elif isinstance(ref, np.ndarray) and ref.shape == ():
            ref = np.ones(self.N) * ref.item()

        
        x = cp.Variable((self.nx, self.N+1))
        u = cp.Variable((self.nu, self.M))
        
        
        objective = 0
        constraints = [x[:, 0] == x0]


        for k in range(self.N):
            objective += (self.Cd @ x[:, k+1] - ref[k]) ** 2
            if k < self.M:
                objective += self.lam * cp.quad_form(u[:, k], np.eye(self.nu))
                constraints += [x[:, k+1] == self.Ad @ x[:, k] + self.Bd @ u[:, k]]
                constraints += [u[:, k] >= 0, u[:, k] <= 1]
            else:
                constraints += [x[:, k+1] == self.Ad @ x[:, k]]
            
        
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        return u[:, 0].value
    
class MPCControllerupp:
    def __init__(self):
        self.A = np.array([
            [-0.3324, -1.0684, 0, 0, 0, 0],
            [1.0000, 0, 0, 0, 0, 0],
            [0, 0, -0.0474, -0.1713, 0, 0],
            [0, 0, 0.2500, 0, 0, 0],
            [0, 0, 0, 0, -0.0014, -0.1980],
            [0, 0, 0, 0, 0.2500, 0]
        ])

        self.B = np.array([
            [0.0625, 0, 0],
            [0, 0, 0],
            [0, 0.5000, 0],
            [0, 0, 0],
            [0, 0, 0.0312],
            [0, 0, 0]
        ])

        self.C = np.array([
            [-0.0401, -0.0308, -0.3769, 0.2964, 0.0192, 0.0181]
        ])

        self.D = np.array([
            [0, 0, 0]
        ])

        self.dt = 0.1      
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.Ad = sp.linalg.expm(self.A*self.dt)
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad-np.eye(self.nx)) @ self.B
        self.Cd = self.C
        self.Dd = self.D
        self.N = 1      
        self.M = 1       
        self.lam = 1e-5  
    
    def controlup(self, x0, ref):
        
        if isinstance(ref, (float, int)):
            ref = np.ones(self.N) * ref
        elif isinstance(ref, np.ndarray) and ref.shape == ():
            ref = np.ones(self.N) * ref.item()

        
        x = cp.Variable((self.nx, self.N+1))
        u = cp.Variable((self.nu, self.M))
        
        
        objective = 0
        constraints = [x[:, 0] == x0]


        for k in range(self.N):
            objective += (self.Cd @ x[:, k+1] - ref[k]) ** 2
            if k < self.M:
                objective += self.lam * cp.quad_form(u[:, k], np.eye(self.nu))
                constraints += [x[:, k+1] == self.Ad @ x[:, k] - self.Bd @ u[:, k]]
                constraints += [u[:, k] >= 0, u[:, k] <= 250]
            else:
                constraints += [x[:, k+1] == -self.Ad @ x[:, k]]
        
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        return u[:, 0].value

file_name = "data_file.data"
file = open(file_name, "w")

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
            scaled.append((val - data_range[0]) * canvas_width / (data_range[1] - data_range[0]))
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
            self.x_range = (self.x_range[0] + x - x_max_visible, self.x_range[1] + x - x_max_visible)
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

class MainWindow(object):
    def __init__(self):
        self.is_running = False
        self.rett = None
        self.root = tkinter.Tk()
        self.left_frame = tkinter.Frame(self.root)
        self.left_frame.pack(side='left')
        self.button = tkinter.Button(self.left_frame, text='launch/stop', command=self.launch_callback)
        self.button.pack()
        self.shm_entry = tkinter.Entry(self.left_frame, width=30)
        self.shm_entry.insert(tkinter.END, 'shm_0')
        self.shm_entry.pack()
        self.time_val = tkinter.StringVar()
        self.time_val.set('time_val')
        self.output_val = tkinter.StringVar()
        self.output_val.set('output')
        self.label = tkinter.Label(self.left_frame, textvariable=self.time_val)
        self.label.pack()
        self.time_label = tkinter.Label(self.left_frame, textvariable=self.output_val)
        self.time_label.pack()
        self.frame = tkinter.Frame(self.root)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame)
        self.canvas.pack()
        self.chart = Chart(self.canvas, (0.0, 5.0), (1e-6,-1e-6))
        self.shm = AmeCommunication.AmeSharedmem()
        self.chrono = Chrono()
        self.last_refresh_time = 0
        self.mpc_controllerup = MPCControllerup()
        self.mpc_controllerdown = MPCControllerdown()
        self.mpc_controllerupp = MPCControllerupp()

        tkinter.mainloop()
        
    def launch_callback(self):
        if self.is_running:
            self.is_running = False
            self.chart.clean()
            self.canvas.after(100, self.shm.close)
        else:
            try:
                self.shm.init(False, str(self.shm_entry.get()), 5, 7)
                ret = self.shm.exchange([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.is_running = True
            except RuntimeError:
                return
            self.chrono.set_time(ret[1])
            self.last_output_dis = ret[2]
            self.last_output_velocity = ret[3]
            self.last_output_dis_target = ret[5]
            self.last_output_velocity_target = ret[6]
            self.add_points()

    def add_points(self):
        while self.is_running:
            t = self.chrono.get_time()
            try:
              
                current_state = np.array([self.last_output_velocity, 0, 0,0,0,0])
                output_val_in3 = self.mpc_controllerupp.controlup(current_state, self.last_output_dis_target)[1]*23
                output_val_up = self.mpc_controllerup.controlup(current_state, self.last_output_dis_target)[0]*1.55
                output_val_down = self.mpc_controllerdown.controldown(current_state, self.last_output_dis_target)[0]*0.52
                

                rett = self.shm.exchange([0.0, t, output_val_in3, 
                                          output_val_down, output_val_up])
                
                self.last_output_dis = rett[2]
                self.last_output_velocity = rett[3]
                self.last_output_dis_target = rett[5]
                self.last_output_velocity_target = rett[6]

                new_point = (rett[1], rett[2], rett[3], rett[5], rett[6])
                file.write(f"{new_point[0]} {new_point[1]} {new_point[2]} {new_point[3]} {new_point[4]}\n")

            except RuntimeError as e:
                print(f"Error during exchange: {e}")
                return

            self.chart.add_point(rett[1], rett[2])
            t = self.chrono.get_time()
            if t - self.last_refresh_time > 0.1:
                print("Warning: exchange rate goes too fast to ensure a smooth display")
                print("Try with a smaller sample time")
                self.last_refresh_time = t
                self.chart.update()
                self.time_val.set('time: ' + str(self.chrono.get_time()))
                self.output_val.set('val: ' + str(self.last_output_dis))
                self.chart.canvas.after(1, self.add_points)
                break
            elif rett[1] - t > 0.001:
                self.last_refresh_time = t
                self.chart.update()
                self.time_val.set('time: ' + str(self.chrono.get_time()))
                self.output_val.set('val: ' + str(self.last_output_dis))
                self.chart.canvas.after(1 + int((rett[1] - t) * 1000), self.add_points)
                break
    def finalize(self):
        file.close()
        self.shm.close()
if __name__ == '__main__':
    main_window = MainWindow()

    main_window.add_points() 

    main_window.finalize()
