import tkinter
import os
from tkinter import *


class BudgetPlan:
    def create_window(self):
        self.window = tkinter.Tk()
        self.window.title("BRASA HACKS: NUBANK")
        # In order to prevent the window from getting resized you will call 'resizable' method on the window
        self.window.resizable(0, 0)

    def budget_screen(self):
        self.window.destroy()
        self.create_window()
        self.window.geometry("500x650")
        button_widget = tkinter.Button(self.window, text="I'm a Button")
        button_widget.pack()
        # pack is used to show the object in the window
        label = tkinter.Label(self.window, text="I'm just a text!").pack()
        self.window.mainloop()

    def init_screen(self):
        # In order to display the image in a GUI, you will use the 'PhotoImage' method of Tkinter.
        # It will an image from the directory (specified path) and store the image in a variable.
        self.create_window()
        icon = tkinter.PhotoImage(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Nubank.JPG'))

        # Finally, to display the image you will make use of the 'Label' method and pass the 'image'
        # variable as a parameter and use the pack() method to display inside the GUI
        label = tkinter.Label(self.window, image=icon)
        label.pack()

        btns_frame = Frame(self.window, bg="grey")
        btns_frame.place(x=0, y=260)

        # label2 = tk.Label(master=frame, text="I'm at (75, 75)", bg="yellow")
        # label2.place(x=75, y=75)

        budget_btn = Button(btns_frame, text="$\tPlanejar finanças", bd=0, bg="white", cursor="hand2",
                            width=32, height=0, font=('arial', 14),
                            command=lambda: self.budget_screen()).pack(side=LEFT, padx=00, pady=0)

        # budget.place(x=0, y=50)
        self.window.mainloop()


if __name__ == "__main__":
    obj = BudgetPlan()
    obj.init_screen()
