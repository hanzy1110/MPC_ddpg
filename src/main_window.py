import numpy as np
import tkinter
import AmeCommunication

from mpc import MPCControllerdown, MPCControllerup, MPCControllerupp
from helpers import Chart, Chrono

DATA_FILE = "logs/data.txt"


class MainWindow(object):
    def __init__(self):
        self.is_running = False
        self.rett = None

        # TKInterINIT
        self.root = tkinter.Tk()
        self.left_frame = tkinter.Frame(self.root)
        self.left_frame.pack(side="left")
        self.button = tkinter.Button(
            self.left_frame, text="launch/stop", command=self.launch_callback
        )
        self.button.pack()

        # Important for AmeCommunication
        self.shm_entry = tkinter.Entry(self.left_frame, width=30)
        self.shm_entry.insert(tkinter.END, "shm_0")
        self.shm_entry.pack()

        #TKInter stuff:
        self.time_val = tkinter.StringVar()
        self.time_val.set("time_val")
        self.output_val = tkinter.StringVar()
        self.output_val.set("output")
        self.label = tkinter.Label(self.left_frame, textvariable=self.time_val)
        self.label.pack()
        self.time_label = tkinter.Label(self.left_frame, textvariable=self.output_val)
        self.time_label.pack()
        self.frame = tkinter.Frame(self.root)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame)
        self.canvas.pack()
        self.chart = Chart(self.canvas, (0.0, 5.0), (1e-6, -1e-6))

        self.shm = AmeCommunication.AmeSharedmem()

        self.chrono = Chrono()
        self.last_refresh_time = 0

        #MPC Control:
        self.mpc_controllerup = MPCControllerup()
        self.mpc_controllerdown = MPCControllerdown()
        self.mpc_controllerupp = MPCControllerupp()

        # Outfile
        self.file = open(DATA_FILE, "w")

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
                current_state = np.array(
                    [self.last_output_velocity, 0, 0, 0, 0, 0]
                )
                # Modify the state accordingly-self.mpc_controllerupp.controlup(current_state, self.last_output_dis_target)[1]*20
                output_val_in3 = (
                    self.mpc_controllerupp.controlup(
                        current_state, self.last_output_dis_target
                    )[1]
                    * 23
                )
                output_val_up = (
                    self.mpc_controllerup.controlup(
                        current_state, self.last_output_dis_target
                    )[0]
                    * 1.55
                )
                output_val_down = (
                    self.mpc_controllerdown.controldown(
                        current_state, self.last_output_dis_target
                    )[0]
                    * 0.52
                )

                # Just Communication with Amesim?
                # Here it returns the output from the rectangle
                # The data files should be read here
                rett = self.shm.exchange(
                    [0.0, t, output_val_in3, output_val_down, output_val_up]
                )

                self.last_output_dis = rett[2]
                self.last_output_velocity = rett[3]
                self.last_output_dis_target = rett[5]
                self.last_output_velocity_target = rett[6]

                new_point = (rett[1], rett[2], rett[3], rett[5], rett[6])
                self.file.write(
                    f"{new_point[0]} {new_point[1]} {new_point[2]} {new_point[3]} {new_point[4]}\n"
                )

            except RuntimeError as e:
                print(f"Error during exchange: {e}")
                return

            # Render points in chart:
            # [ Not useful ]
            self.chart.add_point(rett[1], rett[2])
            t = self.chrono.get_time()
            if t - self.last_refresh_time > 0.1:
                print("Warning: exchange rate goes too fast to ensure a smooth display")
                print("Try with a smaller sample time")
                self.last_refresh_time = t
                self.chart.update()
                self.time_val.set("time: " + str(self.chrono.get_time()))
                self.output_val.set("val: " + str(self.last_output_dis))
                self.chart.canvas.after(1, self.add_points)
                break
            elif rett[1] - t > 0.001:
                self.last_refresh_time = t
                self.chart.update()
                self.time_val.set("time: " + str(self.chrono.get_time()))
                self.output_val.set("val: " + str(self.last_output_dis))
                self.chart.canvas.after(1 + int((rett[1] - t) * 1000), self.add_points)
                break

    def finalize(self):
        self.file.close()
        self.shm.close()
