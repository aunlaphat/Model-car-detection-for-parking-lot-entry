import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

# ฟังก์ชันสำหรับ parse ข้อมูลจากไฟล์ XML
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ev_count = 0
    non_ev_count = 0
    subclass = None

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label == 'EV':
            ev_count += 1
        elif label == 'Non-EV':
            non_ev_count += 1

    metadata = root.find('metadata')
    if metadata is not None:
        tags = metadata.find('tags')
        if tags is not None:
            for tag in tags.findall('tag'):
                if tag is not None:
                    subclass = tag.text
                    break

    return (ev_count, non_ev_count), subclass

# ฟังก์ชันสำหรับโหลดข้อมูลจากโฟลเดอร์
def load_voc_data(data_dir, image_size=(224, 224)):
    images = []
    labels = []
    subclasses = []
    print(f"Loading data from: {data_dir}")

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for xml_file in os.listdir(class_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(class_path, xml_file)
                    (ev_count, non_ev_count), subclass = parse_voc_annotation(xml_path)

                    image_filename = os.path.splitext(xml_file)[0]
                    image_path_jpg = os.path.join(class_path, f"{image_filename}.jpg")
                    image_path_png = os.path.join(class_path, f"{image_filename}.png")

                    img = None
                    if os.path.exists(image_path_jpg):
                        img = load_img(image_path_jpg, target_size=image_size)
                    elif os.path.exists(image_path_png):
                        img = load_img(image_path_png, target_size=image_size)
                    else:
                        print(f"Image not found for {xml_file}")
                        continue

                    img = img_to_array(img) / 255.0
                    images.append(img)

                    if ev_count > 0:
                        labels.append(1)  # 1 สำหรับ EV
                    else:
                        labels.append(0)  # 0 สำหรับ Non-EV

                    if ev_count > 0 and subclass is not None:
                        subclasses.append(subclass)
                    else:
                        subclasses.append('Non-EV')

    return np.array(images), np.array(labels), subclasses

# ฟังก์ชันสำหรับโหลดข้อมูลทั้งหมด
def load_data(data_dir):
    train_images, train_labels_ev, train_subclasses = load_voc_data(os.path.join(data_dir, 'train'))
    val_images, val_labels_ev, val_subclasses = load_voc_data(os.path.join(data_dir, 'valid'))

    le = LabelEncoder()
    le.fit(train_subclasses + val_subclasses)  # เข้ารหัส subclasses ทั้งหมด

    train_subclasses_encoded = le.transform(train_subclasses)
    val_subclasses_encoded = le.transform(val_subclasses)

    print(f"Train images: {len(train_images)}, Train labels: {len(train_labels_ev)}, Train subclasses: {len(train_subclasses_encoded)}")
    print(f"Validation images: {len(val_images)}, Validation labels: {len(val_labels_ev)}, Validation subclasses: {len(val_subclasses_encoded)}")

    return (train_images, (train_labels_ev, train_subclasses_encoded)), (val_images, (val_labels_ev, val_subclasses_encoded)), le

# ฟังก์ชันสำหรับโหลดแมพ subclass
def load_subclass_mapping():
    return {
        'EV': {
            0: 'BYD-Atto-3',
            1: 'BYD-Dolphin',
            2: 'BYD-sealPerformance',
            3: 'MG-4',
            4: 'MG-EP',
            5: 'MG-ZS-EVStandard',
            6: 'neta-v',
            7: 'ORA-Good-Cat',
            8: 'Tesla-3',
            9: 'Tesla-Y',
        },
        'Non-EV': {
            10: 'Ford-Ranger-Rapter-3.0',
            11: 'Honda-HR-V-e_HEV',
            12: 'Isuzu-d-max-hi-lander-1.3',
            13: 'Isuzu-Mu-x',
            14: 'Mitsubishi-Triton',
            15: 'Nissan-Almera',
            16: 'Toyota-Corolla-Cross',
            17: 'Toyota-Fortuner',
            18: 'Toyota-hilus-revo-prerunner',
            19: 'Toyota-yaris-ATIV',
        }
    }

# ฟังก์ชันสำหรับสร้างโมเดลแบบ multi-output
def create_multi_output_model(input_shape=(224, 224, 3), num_subclasses=20):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers[:8]:
        layer.trainable = False
    for layer in base_model.layers[8:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    ev_output = Dense(1, activation='sigmoid', name='ev_output')(x)  # 1 output สำหรับ EV/Non-EV
    subclass_output = Dense(num_subclasses, activation='softmax', name='subclass_output')(x)  # output สำหรับ subclasses

    return models.Model(inputs=base_model.input, outputs=[ev_output, subclass_output])

# ฟังก์ชันสำหรับฝึกโมเดล
def train_multi_output_model(model, train_data, validation_data):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    train_images, (train_labels_ev, train_subclasses_encoded) = train_data
    val_images, (val_labels_ev, val_subclasses_encoded) = validation_data

    model.compile(
        optimizer=optimizer,
        loss={
            'ev_output': 'binary_crossentropy',
            'subclass_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={'ev_output': 1.0, 'subclass_output': 1.5},  # ให้ความสำคัญมากขึ้นกับ subclass_output
        metrics={
            'ev_output': 'accuracy',
            'subclass_output': 'accuracy'
        }
    )

    # ฝึกโมเดล โดยไม่ใช้ class_weight
    history = model.fit(
        train_images,
        {'ev_output': train_labels_ev, 'subclass_output': train_subclasses_encoded},
        validation_data=(val_images, {'ev_output': val_labels_ev, 'subclass_output': val_subclasses_encoded}),
        epochs=40,
        callbacks=[ModelCheckpoint('model14_3_2.keras', save_best_only=True)]
    )

    return history

# ฟังก์ชันสำหรับการประเมินโมเดล
def evaluate_multi_output_model(model, validation_data, subclass_mapping):
    val_images, (val_labels_ev, val_labels_subclass) = validation_data
    predictions = model.predict(val_images)

    ev_predictions = np.round(predictions[0])  # Get EV/Non-EV predictions
    subclass_predictions = np.argmax(predictions[1], axis=1)  # Get subclass predictions

    print("Classification Report (EV/Non-EV):")
    print(classification_report(val_labels_ev, ev_predictions))

    cm_ev = confusion_matrix(val_labels_ev, ev_predictions)
    sns.heatmap(cm_ev, annot=True, fmt='d')
    plt.title('Confusion Matrix for EV Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("Classification Report (Subclasses):")
    print(classification_report(val_labels_subclass, subclass_predictions))

    cm_subclass = confusion_matrix(val_labels_subclass, subclass_predictions)
    sns.heatmap(cm_subclass, annot=True, fmt='d', xticklabels=subclass_mapping['EV'].keys(), yticklabels=subclass_mapping['Non-EV'].keys())
    plt.title('Confusion Matrix for Subclass Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # เพิ่มส่วนการพิมพ์ subclass ที่คาดการณ์
    for i, pred in enumerate(subclass_predictions):
        if pred in subclass_mapping['EV']:
            print(f"Predicted subclass for image {i}: {subclass_mapping['EV'][pred]} (EV)")
        else:
            print(f"Predicted subclass for image {i}: {subclass_mapping['Non-EV'][pred]} (Non-EV)")

# ฟังก์ชันสำหรับตรวจสอบ subclass ใน XML และความถูกต้องของโครงสร้างไฟล์
def check_annotations(data_dir):
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for xml_file in os.listdir(class_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(class_path, xml_file)
                    (ev_count, non_ev_count), subclass = parse_voc_annotation(xml_path)
                    print(f"{xml_file}: EV count: {ev_count}, Non-EV count: {non_ev_count}, Subclass: {subclass}")

def verify_data_structure(data_dir):
    missing_files = []
    found_subclasses = set()

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for xml_file in os.listdir(class_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(class_path, xml_file)
                    (ev_count, non_ev_count), subclass = parse_voc_annotation(xml_path)

                    image_filename = os.path.splitext(xml_file)[0]
                    image_path_jpg = os.path.join(class_path, f"{image_filename}.jpg")
                    image_path_png = os.path.join(class_path, f"{image_filename}.png")

                    if not os.path.exists(image_path_jpg) and not os.path.exists(image_path_png):
                        missing_files.append(xml_file)

                    if subclass:
                        found_subclasses.add(subclass)

    return missing_files, found_subclasses

# ฟังก์ชันสำหรับแสดงกราฟความแม่นยำและความสูญเสีย
def plot_training_history(history):
    # กราฟความแม่นยำ
    plt.figure(figsize=(12, 5))

    # ความแม่นยำในการฝึก
    plt.subplot(1, 2, 1)
    plt.plot(history.history['ev_output_accuracy'], label='Train EV Accuracy')
    plt.plot(history.history['val_ev_output_accuracy'], label='Val EV Accuracy')
    plt.plot(history.history['subclass_output_accuracy'], label='Train Subclass Accuracy')
    plt.plot(history.history['val_subclass_output_accuracy'], label='Val Subclass Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # กราฟความสูญเสีย
    plt.subplot(1, 2, 2)
    plt.plot(history.history['ev_output_loss'], label='Train EV Loss')
    plt.plot(history.history['val_ev_output_loss'], label='Val EV Loss')
    plt.plot(history.history['subclass_output_loss'], label='Train Subclass Loss')
    plt.plot(history.history['val_subclass_output_loss'], label='Val Subclass Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# ฟังก์ชันหลักสำหรับตรวจสอบโครงสร้างข้อมูล
def main():
    data_dir = r'C:\Users\USER\Desktop\Pattern_G2_code\car Detection 3.v3i.voc'
    subclass_mapping = load_subclass_mapping()
    
    # ตรวจสอบโฟลเดอร์ train
    print("ตรวจสอบโฟลเดอร์ train...")
    train_dir = os.path.join(data_dir, 'train')
    train_missing_files, train_found_subclasses = verify_data_structure(train_dir)
    
    # ตรวจสอบโฟลเดอร์ valid
    print("ตรวจสอบโฟลเดอร์ valid...")
    valid_dir = os.path.join(data_dir, 'valid')
    valid_missing_files, valid_found_subclasses = verify_data_structure(valid_dir)

    # รายงานผลการตรวจสอบ
    if train_missing_files:
        print(f"พบไฟล์ XML ใน train ที่ไม่มีภาพที่เกี่ยวข้อง: {train_missing_files}")
    else:
        print("ไม่พบปัญหาไฟล์ในโฟลเดอร์ train")

    if valid_missing_files:
        print(f"พบไฟล์ XML ใน valid ที่ไม่มีภาพที่เกี่ยวข้อง: {valid_missing_files}")
    else:
        print("ไม่พบปัญหาไฟล์ในโฟลเดอร์ valid")

    # แสดง subclass ที่พบ
    all_found_subclasses = train_found_subclasses.union(valid_found_subclasses)
    print(f"Subclass ที่พบทั้งหมด: {all_found_subclasses}")
    num_subclasses = len(all_found_subclasses)  # ตั้งค่า num_subclasses ตามจำนวนที่พบ
    print(f"จำนวน subclass ที่พบ: {num_subclasses}")

    # โหลดข้อมูลสำหรับการฝึก
    train_data, val_data, label_encoder = load_data(data_dir)  

    # สร้างโมเดล multi-output โดยใช้ num_subclasses
    model = create_multi_output_model(num_subclasses=len(subclass_mapping['EV']) + len(subclass_mapping['Non-EV']))
    history = train_multi_output_model(model, train_data, val_data)

    evaluate_multi_output_model(model, val_data, subclass_mapping)
    plot_training_history(history)

if __name__ == "__main__":
    main()
