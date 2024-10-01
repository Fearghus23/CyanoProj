# inference.py
import torch
from model import create_model
import cv2
import torchvision.transforms as T
import sqlite3
import os

def load_model(num_classes, model_path):
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def detect_and_count(model, image_path, classes, confidence_threshold=0.8):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    outputs = outputs[0]
    detections = []
    counts = {cls: 0 for cls in classes}
    for score, label in zip(outputs['scores'], outputs['labels']):
        if score >= confidence_threshold:
            cls_name = classes[label.item()]
            counts[cls_name] += 1
    return counts

def save_to_database(image_id, counts):
    conn = sqlite3.connect('database/counts.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS counts (
                        image_id TEXT PRIMARY KEY,
                        bacteria_counts TEXT)''')
    counts_str = ','.join(f'{k}:{v}' for k, v in counts.items())
    cursor.execute('INSERT OR REPLACE INTO counts (image_id, bacteria_counts) VALUES (?, ?)', (image_id, counts_str))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    classes = ['Anabaena', 'Aphanizomenon', 'Chroococcales', 'Cylindrospermopsis', 'Dolichospermum', 'Microcystis', 'Nostoc', 'Oscillatoria', 'Phormidium', 'Planktothrix', 'Raphidiopsis', 'Rivularia', 'Synechococcus']
    num_classes = len(classes)
    model = load_model(num_classes, 'models/efficientdet.pth')
    test_images_dir = 'data/test'
    for img_name in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_name)
        counts = detect_and_count(model, img_path, classes)
        save_to_database(img_name, counts)
