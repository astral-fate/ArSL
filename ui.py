import tkinter as tk  #  مكتبة Tkinter لإنشاء الواجهة الرسومية
from tkinter import filedialog  #  filedialog لفتح الملفات
from PIL import Image, ImageTk  #  لتحميل الصور وعرضها داخل التطبيق
import os  #  الملفات

# إنشاء نافذة التطبيق
root = tk.Tk()
root.title("Arabic Sign Language Translator")  # عنوان النافذة
root.geometry("800x500")  # ضبط أبعاد النافذة
root.configure(bg="#f2f7f9")  #  لون خلفية النافذة أبيض مائل للأزرق

# إعداد الخطوط والألوان
title_font = ("Segoe UI", 24, "bold")  # خط عنوان
subtitle_font = ("Segoe UI", 14)  # خط فرعي
text_font = ("Segoe UI", 12)  # خط النصوص
button_font = ("Segoe UI", 12, "bold")  # خط الأزرار

# نظام ألوان محسن وأكثر راحة للعين
main_green = "#5caf99"  # لون أخضر معتدل
accent_blue = "#5d8ba7"  # لون أزرق متوسط
bg_gradient = "#f2f7f9"  # خلفية بيضاء زرقاء فاتحة
text_color = "#2c3e50"  # لون النصوص الرئيسية معتدل
hover_color = "#3e8e75"  # لون عند تمرير الماوس على الأزرار

# إضافة إطار رئيسي مع تأثير ظل خفيف
main_frame = tk.Frame(root, bg=bg_gradient, bd=1, relief=tk.RIDGE)
main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.92, relheight=0.92)

# إضافة شريط علوي بلون أخضر
header_frame = tk.Frame(main_frame, bg=main_green, height=8)
header_frame.pack(fill=tk.X)

# إضافة عنوان رئيسي مع خط حديث
title_label = tk.Label(main_frame, text="Arabic Sign Language Translator",
                      font=title_font, bg=bg_gradient, fg=text_color)
title_label.pack(pady=20)  # وضع العنوان مع إضافة مسافة للأسفل

# إضافة وصف للمستخدم
description = "Begin live translation using your camera, or upload an image or video file!"
desc_label = tk.Label(main_frame, text=description, font=subtitle_font,
                     bg=bg_gradient, fg=text_color, wraplength=600, justify="center")
desc_label.pack(pady=10)  # وضع النص في منتصف الشاشة 

# إضافة متغير لحالة التطبيق
status_text = tk.StringVar()
status_text.set("Ready to translate...")

# دالة فتح الكاميرا (ترجمة مباشرة)
def open_camera():

    # تغيير رسالة الحالة
    status_text.set("Camera feature coming soon...")
    # إظهار الأيقونة وتحديث اللون
    status_icon.config(text="⏳")
    status_label.config(fg="#e67e22")  # لون برتقالي للإشعار

# دالة تحميل صورة من الجهاز
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        # طباعة مسار الملف
        print(f"Image selected: {file_path}")
        # تغيير حالة التطبيق
        status_text.set("Image analysis feature coming soon...")
        # إظهار الأيقونة وتحديث اللون
        status_icon.config(text="⏳")
        status_label.config(fg="#e67e22")  # لون برتقالي للإشعار

# التحقق من وجود أيقونات الأزرار
def load_icon(icon_name, size=(50, 50)):
    """ تحميل الأيقونات مع التحقق من وجودها """
    if os.path.exists(icon_name):
        img = Image.open(icon_name).resize(size)
        return ImageTk.PhotoImage(img)
    else:
        print(f"⚠️ Warning: {icon_name} not found! Using text button instead.")
        return None

# تحميل الأيقونات
camera_photo = load_icon("camera_icon.png")
upload_photo = load_icon("upload_icon.png")

# إنشاء إطار للأزرار مع تصميم بسيط
button_frame = tk.Frame(main_frame, bg=bg_gradient)
button_frame.pack(pady=30)

# خصائص الأزرار المحسنة للمظهر والاستخدام
button_props = {
    'bg': main_green,
    'fg': 'white',
    'activebackground': hover_color,
    'activeforeground': 'white',
    'padx': 25,
    'pady': 12,
    'bd': 0,
    'relief': tk.FLAT,
    'borderwidth': 0,
    'highlightthickness': 0
}

# زر فتح الكاميرا للترجمة المباشرة - تصميم محسن
if camera_photo:
    camera_button = tk.Button(button_frame, image=camera_photo, command=open_camera, **button_props)
else:
    camera_button = tk.Button(button_frame, text="📷 Camera", command=open_camera,
                             font=button_font, **button_props, width=10)

camera_button.grid(row=0, column=0, padx=25, pady=15)  # وضع الزر في مكان مناسب

# زر رفع الصورة من الجهاز - تصميم محسن
if upload_photo:
    upload_button = tk.Button(button_frame, image=upload_photo, command=upload_image, **button_props)
else:
    upload_button = tk.Button(button_frame, text="⬆ Upload", command=upload_image,
                             font=button_font, **button_props, width=10)

upload_button.grid(row=0, column=1, padx=25, pady=15)  # وضع الزر بجانب زر الكاميرا

# إضافة إطار لرسالة الحالة
status_frame = tk.Frame(main_frame, bg=bg_gradient)
status_frame.pack(pady=20)

# إضافة أيقونة الحالة
status_icon = tk.Label(status_frame, text="🔍", font=("Segoe UI", 16), bg=bg_gradient)
status_icon.pack(side=tk.LEFT, padx=(0, 5))

# إضافة ملصق لعرض حالة التطبيق
status_label = tk.Label(status_frame, textvariable=status_text, font=text_font,
                       bg=bg_gradient, fg=text_color)
status_label.pack(side=tk.LEFT)

# إضافة شريط معلومات سفلي
footer_frame = tk.Frame(main_frame, bg=bg_gradient, pady=10)
footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

# إضافة معلومات الإصدار
version_label = tk.Label(footer_frame, text="Version 1.0", font=("Segoe UI", 8),
                        bg=bg_gradient, fg="#95a5a6")
version_label.pack(side=tk.RIGHT, padx=10)

# إضافة شريط سفلي بلون أزرق
bottom_frame = tk.Frame(main_frame, bg=accent_blue, height=8)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# تشغيل التطبيق
root.mainloop()
