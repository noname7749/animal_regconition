import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.keras.models import load_model
model = load_model('./model1_catsVSdogs.h5')

classes = {
    0:'its a cat',
    1:'its a dog',
}

top=tk.Tk()
top.geometry('800x600')
top.title('CatsVSDogs Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((128,128))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image/255
    # Thay đổi cách dự đoán ở đây
    pred = numpy.argmax(model.predict(image), axis=-1)[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign)

# Phần code còn lại giữ nguyên
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
                        command=lambda: classify(file_path),
                        padx=10,pady=5)
    classify_b.configure(background='#364156',
                        foreground='white',
                        font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="CatsVSDogs Classification",
                pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()