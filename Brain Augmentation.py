#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import zipfile
from PIL import Image
import numpy as np
import albumentations as A
import cv2

# ---------------------------
# 1) فك الضغط
# ---------------------------
zip_path = r"C:\Users\ms\Downloads\Tumor.zip"
extract_path = r"C:\Users\ms\Downloads"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("تم فك الضغط بنجاح.")

# ---------------------------
# 2) تحويل كل الصور إلى PNG
# ---------------------------
def convert_to_png(folder):
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path): 
            continue
        
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            try:
                img = Image.open(img_path)
                img = img.convert("RGB")

                new_name = os.path.splitext(img_name)[0] + ".png"
                new_path = os.path.join(cls_path, new_name)

                img.save(new_path, "PNG")

                # احذف الصورة القديمة لو الامتداد غير PNG
                if not img_name.lower().endswith(".png"):
                    os.remove(img_path)

            except:
                print(f"خطأ في صورة: {img_path}")

    print("✔️ تم تحويل كل الصور لـ PNG")


convert_to_png(extract_path)


# In[8]:


import os
from PIL import Image

dataset_path = r"C:\Users\ms\Downloads\Stroke"   # غيّر اسم المجلد لو مختلف

def convert_all_to_png(folder):
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue

        print(f"\n📌 Class: {cls}")

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            # تخطي أي ملف مش صورة
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                img = Image.open(img_path).convert("RGB")

                # اسم جديد بصيغة PNG
                new_name = os.path.splitext(img_name)[0] + ".png"
                new_path = os.path.join(cls_path, new_name)

                # حفظ الصورة
                img.save(new_path, "PNG")

                # لو الامتداد الأصلي مش PNG → احذفه
                if not img_name.lower().endswith(".png"):
                    os.remove(img_path)

            except Exception as e:
                print(f"❌ مشكلة في الصورة: {img_path} | {e}")

    print("\n✔️ تم تحويل كل الصور إلى PNG بنجاح!")

convert_all_to_png(dataset_path)


# In[10]:


get_ipython().system('pip install tqdm')


# In[12]:


import os
import cv2
import albumentations as A


# -----------------------------
#  تحديد المجلد الأساسي
# -----------------------------
base_path = r"C:\Users\ms\Downloads\Tumor\Training"

# -----------------------------
#  الأعداد المطلوبة لكل class
# -----------------------------
targets = {
    "glioma": 6500,
    "meningioma": 3200,
    "notumor": 2500,
    "pituitary": 2500
}

# -----------------------------
#  الـ Augmentation Pipeline
# -----------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=25, p=0.5),
])

# -----------------------------
#  دالة لعمل Augmentation
# -----------------------------
def augment_class(folder, target_count):
    images = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    current = len(images)

    print(f"\n📌 Class: {os.path.basename(folder)} | الموجود: {current} | المطلوب: {target_count}")

    idx = 0

    while current < target_count:
        img_name = images[idx % len(images)]
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = augment(image=img)["image"]

        save_name = f"aug_{current}.png"
        save_path = os.path.join(folder, save_name)

        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        current += 1
        idx += 1

    print(f"✔️ انتهى — العدد النهائي: {current}")

# -----------------------------
#  تنفيذ الـ augmentation على كل class
# -----------------------------
for cls, target in targets.items():
    folder = os.path.join(base_path, cls)
    if os.path.isdir(folder):
        augment_class(folder, target)
    else:
        print(f"⚠️ المجلد غير موجود: {folder}")

print("\n🎉 تم إكمال الـ Augmentation لكل الفئات بنجاح!")


# In[13]:


import os
import cv2
import albumentations as A

# -----------------------------
#  تحديد المجلد الأساسي
# -----------------------------
base_path = r"C:\Users\ms\Downloads\Stroke"

# -----------------------------
#  الأعداد المطلوبة لكل class
# -----------------------------
targets = {
    "StrokeMR": 5000,
    "NormalMR": 2500
}

# -----------------------------
#  الـ Augmentation Pipeline
# -----------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=25, p=0.5),
])

# -----------------------------
#  دالة لعمل Augmentation
# -----------------------------
def augment_class(folder, target_count):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current = len(images)

    print(f"\n📌 Class: {os.path.basename(folder)} | الموجود: {current} | المطلوب: {target_count}")

    idx = 0

    while current < target_count:
        img_name = images[idx % len(images)]
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = augment(image=img)["image"]

        save_name = f"aug_{current}.png"
        save_path = os.path.join(folder, save_name)

        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        current += 1
        idx += 1

    print(f"✔️ انتهى — العدد النهائي: {current}")

# -----------------------------
#  تنفيذ الـ augmentation على كل class
# -----------------------------
for cls, target in targets.items():
    folder = os.path.join(base_path, cls)
    if os.path.isdir(folder):
        augment_class(folder, target)
    else:
        print(f"⚠️ المجلد غير موجود: {folder}")

print("\n🎉 تم إكمال الـ Augmentation لكل الفئات بنجاح!")


# In[15]:


import os
import cv2
import albumentations as A

# -----------------------------
#  تحديد المجلد الأساسي
# -----------------------------
base_path = r"C:\Users\ms\Downloads\Alzheimer's"

# -----------------------------
#  الأعداد المطلوبة لكل class
# -----------------------------
targets = {
    "Mild Impairment": 3800,
    "Moderate Impairment": 4200,
    "No Impairment": 2500,
    "Very Mild Impairment": 3400
}

# -----------------------------
#  الـ Augmentation Pipeline
# -----------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=25, p=0.5),
])

# -----------------------------
#  دالة لعمل Augmentation
# -----------------------------
def augment_class(folder, target_count):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current = len(images)

    print(f"\n📌 Class: {os.path.basename(folder)} | الموجود: {current} | المطلوب: {target_count}")

    idx = 0

    while current < target_count:
        img_name = images[idx % len(images)]
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = augment(image=img)["image"]

        save_name = f"aug_{current}.png"
        save_path = os.path.join(folder, save_name)

        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        current += 1
        idx += 1

    print(f"✔️ انتهى — العدد النهائي: {current}")

# -----------------------------
#  تنفيذ الـ augmentation على كل class
# -----------------------------
for cls, target in targets.items():
    folder = os.path.join(base_path, cls)
    if os.path.isdir(folder):
        augment_class(folder, target)
    else:
        print(f"⚠️ المجلد غير موجود: {folder}")

print("\n🎉 تم إكمال الـ Augmentation لكل الفئات بنجاح!")


# In[16]:


import os
import cv2
import albumentations as A

# -----------------------------
#  تحديد المجلد الأساسي
# -----------------------------
folder = r"C:\Users\ms\Downloads\Brain Dataset\Epilepsy"

# -----------------------------
#  العدد المطلوب
# -----------------------------
target_count = 3000

# -----------------------------
#  الـ Augmentation Pipeline
# -----------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=25, p=0.5),
])

# -----------------------------
#  دالة لعمل Augmentation
# -----------------------------
def augment_class(folder, target_count):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current = len(images)

    print(f"\n📌 Class: {os.path.basename(folder)} | الموجود: {current} | المطلوب: {target_count}")

    idx = 0

    while current < target_count:
        img_name = images[idx % len(images)]
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = augment(image=img)["image"]

        save_name = f"aug_{current}.png"
        save_path = os.path.join(folder, save_name)

        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        current += 1
        idx += 1

    print(f"✔️ انتهى — العدد النهائي: {current}")

# -----------------------------
#  تنفيذ الـ augmentation
# -----------------------------
if os.path.isdir(folder):
    augment_class(folder, target_count)
else:
    print(f"⚠️ المجلد غير موجود: {folder}")

print("\n🎉 تم إكمال الـ Augmentation لفئة Epilepsy بنجاح!")


# In[ ]:




