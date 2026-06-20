import os
import sys
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import segmentation_models_pytorch as smp

# PyQt5 UI Framework
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTabWidget, QSplitter, 
    QListWidget, QListWidgetItem, QProgressBar, QGroupBox, QGridLayout,
    QStackedWidget, QMessageBox, QFrame, QLineEdit, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QColor, QPalette, QBrush

# Import OD model architecture
try:
    from od_mimari import get_segmentation_model
except ImportError:
    QMessageBox.critical(None, "Hata", "od_mimari.py dosyası bulunamadı. Lütfen model mimarisinin aynı dizinde olduğundan emin olun.")
    sys.exit(1)

# --- SİSTEM VE CİHAZ AYARLARI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL YOLLARI VE VERİ SETİ AYARLARI ---
BASE_DIR = r"C:\Users\Dogan\Desktop\Ayberk Dosyalar\Bitirme Projesi\Fundus\FundusProje"
yolo_path = os.path.join(BASE_DIR, "Macula_Detect", "runs", "detect", "yolov8_macula_bebek", "weights", "best.pt")
od_model_path = os.path.join(BASE_DIR, "Final-Process_Son_Okul_Deneme", "od_bebek_model.pth")
vessel_model_path = os.path.join(BASE_DIR, "Vessel_Segment", "vessel_bebek_model.pth")
test_img_dir = os.path.join(BASE_DIR, "Final-Process_Son_Okul_Deneme", "vessel_segment_bebek_data1", "train", "image")
test_mask_dir = os.path.join(BASE_DIR, "Final-Process_Son_Okul_Deneme", "vessel_segment_bebek_data1", "train", "mask")

# --- Transform Ayarları ---
od_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

vessel_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# --- MANUAL OPTIC DISK & MACULA BULUCU (FALLBACK) ---
def manual_fallback_process(image_rgb, od_model, transform, device):
    h, w = image_rgb.shape[:2]
    input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = od_model(input_tensor)
        prediction = (output > 0.5).cpu().numpy().squeeze()
    
    od_mask = cv2.resize(prediction.astype(np.uint8), (w, h))
    contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] == 0: return None
    cx_od, cy_od = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    od_radius = int(np.sqrt(cv2.contourArea(cnt) / np.pi))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    if not cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]: return None
    fov_mask_clean = cv2.erode(fov_mask, np.ones((51, 51), np.uint8)) 

    y, x = np.indices((h, w))
    dist_from_od = np.sqrt((x - cx_od)**2 + (y - cy_od)**2)
    mask_distance = (dist_from_od > od_radius * 2.0) & (dist_from_od < od_radius * 6.0)

    green_ch = image_rgb[:, :, 1]
    green_blurred = cv2.GaussianBlur(green_ch, (51, 51), 0)
    inverted = cv2.bitwise_not(green_blurred)
    
    inverted[fov_mask_clean == 0] = 0
    inverted[~mask_distance] = 0
    inverted[od_mask > 0] = 0

    valid_px = inverted[inverted > 0]
    if len(valid_px) > 0:
        thresh = np.percentile(valid_px, 99.5) 
        _, bw = cv2.threshold(inverted, thresh, 255, cv2.THRESH_BINARY)
        m_contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if m_contours:
            best_m_cnt = max(m_contours, key=cv2.contourArea)
            M_m = cv2.moments(best_m_cnt)
            if M_m["m00"] != 0:
                macula_center = (int(M_m["m10"] / M_m["m00"]), int(M_m["m01"] / M_m["m00"]))
            else:
                _, _, _, macula_center = cv2.minMaxLoc(inverted)
        else:
            _, _, _, macula_center = cv2.minMaxLoc(inverted)
    else:
        return None

    return {"od_center": (cx_od, cy_od), "macula_center": macula_center, "od_radius": od_radius}


# --- GEOMETRİK BÖLGELEME VE DAMAR SAYIM FONKSİYONU ---
def process_vessel_quadrants_and_draw(image_rgb, od_center, macula_center, vessel_mask, od_radius, title_text=""):
    h, w = image_rgb.shape[:2]
    out_img = image_rgb.copy()
    
    cx, cy = map(float, od_center)
    mx, my = map(float, macula_center)
    
    is_right_eye = bool(mx < cx)
    
    # 1. GEOMETRİK DOĞRULARIN HESAPLANMASI
    dx = mx - cx
    dy = my - cy
    if dx == 0 and dy == 0: dx = 1.0
    
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / length, dy / length
    perpend_x, perpend_y = -uy, ux
    
    diagonal = float(np.sqrt(h**2 + w**2))
    
    # Doğruları çiz
    cv2.line(out_img, (int(cx - ux * diagonal), int(cy - uy * diagonal)), (int(cx + ux * diagonal), int(cy + uy * diagonal)), (255, 0, 0), 2)
    cv2.line(out_img, (int(cx - perpend_x * diagonal), int(cy - perpend_y * diagonal)), (int(cx + perpend_x * diagonal), int(cy + perpend_y * diagonal)), (0, 255, 255), 2)

    # OD Etrafında Beyaz Sayım Çemberi
    count_radius = int(od_radius * 2.5) if od_radius > 0 else 60
    cv2.circle(out_img, (int(cx), int(cy)), count_radius, (255, 255, 255), 2)
    
    # 2. PROJEKSİYON TABANLI BÖLGELEME MOTORU
    y_indices, x_indices = np.indices((h, w))
    
    # Kırmızı çizginin üstü/altı (Anatomik Superior/Inferior)
    if mx - cx != 0:
        m1 = dy / dx
        c1 = cy - m1 * cx
        above_red = y_indices < (m1 * x_indices + c1)
    else:
        above_red = x_indices > cx if is_right_eye else x_indices < cx

    # İzdüşüm vektör skaler değeri
    dot_product = (x_indices - cx) * ux + (y_indices - cy) * uy

    # 3. DİNAMİK KADRAN VE METİN KONUMU YAPILANDIRMASI
    quadrants = {}
    text_config = []
    
    is_temporal = dot_product > 0

    if is_right_eye:  # --- SAĞ GÖZ ---
        quadrants["Superior Temporal"] = above_red & is_temporal
        quadrants["Superior Nasal"]    = above_red & ~is_temporal
        quadrants["Inferior Temporal"] = ~above_red & is_temporal
        quadrants["Inferior Nasal"]    = ~above_red & ~is_temporal
        
        text_config = [
            ("Superior Temporal", (20, 40)),        # Sol Üst
            ("Superior Nasal", (w - 320, 40)),      # Sağ Üst
            ("Inferior Temporal", (20, h - 30)),     # Sol Alt
            ("Inferior Nasal", (w - 320, h - 30))     # Sağ Alt
        ]
    else:             # --- SOL GÖZ ---
        quadrants["Superior Nasal"]    = above_red & ~is_temporal
        quadrants["Superior Temporal"] = above_red & is_temporal
        quadrants["Inferior Nasal"]    = ~above_red & ~is_temporal
        quadrants["Inferior Temporal"] = ~above_red & is_temporal
        
        text_config = [
            ("Superior Nasal", (20, 40)),           # Sol Üst
            ("Superior Temporal", (w - 320, 40)),   # Sağ Üst
            ("Inferior Nasal", (20, h - 30)),        # Sol Alt
            ("Inferior Temporal", (w - 320, h - 30))   # Sağ Alt
        ]

    # Damarları Renklendir (Arter: Kırmızı, Ven: Mavi)
    out_img[vessel_mask == 1] = [255, 0, 0] 
    out_img[vessel_mask == 2] = [0, 0, 255] 

    # Çember Çeper İzolasyonu (Arter ve Ven sayımları bu çember çeperi üzerindeki damarlar içindir)
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring_mask, (int(cx), int(cy)), count_radius + 2, 255, -1)
    cv2.circle(ring_mask, (int(cx), int(cy)), count_radius - 2, 0, -1)
    ring_mask_bool = ring_mask > 0

    # Bölge Sayımları
    results = {}
    for q_name in ["Superior Nasal", "Superior Temporal", "Inferior Nasal", "Inferior Temporal"]:
        q_mask = quadrants[q_name]
        
        q_artery = ((vessel_mask == 1) & q_mask & ring_mask_bool).astype(np.uint8)
        q_vein   = ((vessel_mask == 2) & q_mask & ring_mask_bool).astype(np.uint8)
        
        num_arteries = int(cv2.connectedComponentsWithStats(q_artery)[0])
        num_veins    = int(cv2.connectedComponentsWithStats(q_vein)[0])
        
        final_a = (num_arteries - 1) if num_arteries > 0 else 0
        final_v = (num_veins - 1) if num_veins > 0 else 0
        
        results[q_name] = {"A": max(0, final_a), "V": max(0, final_v)}

    # Dinamik olarak yazıları çiz
    for q_name, pos in text_config:
        count_str = f"{q_name}: A:{results[q_name]['A']} V:{results[q_name]['V']}"
        cv2.putText(out_img, count_str, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.circle(out_img, (int(cx), int(cy)), 8, (0, 255, 0), -1)       
    cv2.circle(out_img, (int(mx), int(my)), 8, (0, 255, 255), -1) 

    cv2.putText(out_img, title_text, (w // 2 - 100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    eye_info = "Sag Goz (Right Eye)" if is_right_eye else "Sol Goz (Left Eye)"
    return out_img, eye_info


# --- QPIXMAP ÇEVİRİCİ ---
def cv2_to_qpixmap(cv_img):
    cv_img = cv_img.copy()
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


# --- ÇIFT RESIM SÜRÜKLE BIRAK ETİKETİ ---
class DropZoneLabel(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("📸 Bebek Fundus Resmini Sürükleyin\nveya Seçmek için Tıklayın")
        self.setObjectName("DropZone")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            main_win = self.window()
            if hasattr(main_win, 'select_file'):
                main_win.select_file()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("active", True)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("active", False)
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("active", False)
        self.style().polish(self)
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.fileDropped.emit(file_path)
                break


# --- DİNAMİK RESİM BOYUTLANDIRICI ETİKET ---
class ScaledImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pix = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pix):
        self.pix = pix
        self.update_image()

    def update_image(self):
        if self.pix:
            scaled_pix = self.pix.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pix)
        else:
            super().setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()


# --- ARKA PLAN MODEL YÜKLEYİCİ ---
class ModelLoaderWorker(QThread):
    loaded = pyqtSignal(object, object, object)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def run(self):
        try:
            self.status.emit("YOLOv8 Makula ağırlıkları yükleniyor...")
            yolo_model = YOLO(yolo_path)
            
            self.status.emit("Unet++ Optik Disk modeli kuruluyor...")
            od_model = get_segmentation_model().to(device)
            od_model.load_state_dict(torch.load(od_model_path, map_location=device))
            od_model.eval()

            self.status.emit("Unet++ EfficientNet-B5 Damar modeli yükleniyor...")
            vessel_model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b5",
                encoder_weights=None, 
                in_channels=3,
                classes=3,
            ).to(device)
            vessel_model.load_state_dict(torch.load(vessel_model_path, map_location=device))
            vessel_model.eval()

            self.status.emit("Tüm modeller başarıyla yüklendi!")
            self.loaded.emit(yolo_model, od_model, vessel_model)
        except Exception as e:
            self.error.emit(str(e))


# --- ARKA PLAN HİBRİT ANALİZ EKSPERTİZİ ---
class AnalysisWorker(QThread):
    progress = pyqtSignal(str, int)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, img_path, yolo_model, od_model, vessel_model):
        super().__init__()
        self.img_path = img_path
        self.yolo_model = yolo_model
        self.od_model = od_model
        self.vessel_model = vessel_model

    def run(self):
        try:
            self.progress.emit("Fundus görüntüsü okunuyor...", 10)
            img_bgr = cv2.imread(self.img_path)
            if img_bgr is None:
                raise ValueError(f"Resim dosyası açılamadı: {self.img_path}")
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            base_name = os.path.splitext(os.path.basename(self.img_path))[0]
            
            # 1. Ground Truth Maskesi Arama
            self.progress.emit("Referans (Ground Truth) etiketleri aranıyor...", 25)
            vessel_mask_gt = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.PNG', '.JPG', '.JPEG', '.TIF']:
                temp_mask_path = os.path.join(test_mask_dir, base_name + ext)
                if os.path.exists(temp_mask_path):
                    mask_gt_bgr = cv2.imread(temp_mask_path)
                    if mask_gt_bgr is not None:
                        vessel_mask_gt = mask_gt_bgr[:,:,0]
                        vessel_mask_gt = cv2.resize(vessel_mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                        break
            
            # 2. Optik Disk Segmentasyonu
            self.progress.emit("Unet++ ile Optik Disk / Fovea konumu analiz ediliyor...", 45)
            od_res = manual_fallback_process(img_rgb, self.od_model, od_transform, device)
            if od_res:
                od_center = od_res["od_center"]
                od_radius = od_res["od_radius"]
                fallback_macula = od_res["macula_center"]
            else:
                od_center = (int(w * 0.35), int(h * 0.5))
                od_radius = 40
                fallback_macula = (int(w * 0.65), int(h * 0.5))
            
            # 3. YOLOv8 Makula Algılama
            self.progress.emit("YOLOv8 AI ile Makula yapısı taranıyor...", 65)
            yolo_results = self.yolo_model.predict(source=self.img_path, conf=0.1, imgsz=640, verbose=False, augment=True)
            if len(yolo_results[0].boxes) > 0:
                confidences = yolo_results[0].boxes.conf
                max_conf_idx = confidences.argmax().item()
                highest_box = yolo_results[0].boxes[max_conf_idx]
                xmin, ymin, xmax, ymax = map(int, highest_box.xyxy[0])
                macula_center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
                detection_type = f"YOLOv8 AI ({confidences[max_conf_idx]:.2f} Güven)"
            else:
                macula_center = fallback_macula
                detection_type = "Manuel Fallback (Görüntü Analitiği)"
            
            # 4. Damar Segmentasyonu (Unet++ EfficientNet-B5)
            self.progress.emit("Damar katmanları (Arter/Ven) segmentasyonu yapılıyor...", 80)
            v_input = vessel_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                v_output = self.vessel_model(v_input)
                v_pred = torch.argmax(v_output, dim=1).cpu().numpy().squeeze().astype(np.uint8)
            vessel_mask_pred = cv2.resize(v_pred, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 5. Geometrik Bölgeleme Çizimleri
            self.progress.emit("Kadran sınırları ve damar sayım matrisi çiziliyor...", 95)
            
            cx, cy = map(float, od_center)
            mx, my = map(float, macula_center)
            is_right_eye = bool(mx < cx)
            eye_info = "Sağ Göz (Right Eye)" if is_right_eye else "Sol Göz (Left Eye)"
            
            # Görsel overlay oluştur
            final_prediction_view, _ = process_vessel_quadrants_and_draw(
                img_rgb, od_center, macula_center, vessel_mask_pred, od_radius, title_text="MODEL PREDICTION"
            )
            
            if vessel_mask_gt is not None:
                final_gt_view, _ = process_vessel_quadrants_and_draw(
                    img_rgb, od_center, macula_center, vessel_mask_gt, od_radius, title_text="GROUND TRUTH"
                )
            else:
                final_gt_view = None

            # Sadece Damarların Olduğu Maske Çıktıları
            vessel_only_pred = np.zeros((h, w, 3), dtype=np.uint8)
            vessel_only_pred[vessel_mask_pred == 1] = [255, 0, 0] # Arter: Kırmızı
            vessel_only_pred[vessel_mask_pred == 2] = [0, 0, 255] # Ven: Mavi

            if vessel_mask_gt is not None:
                vessel_only_gt = np.zeros((h, w, 3), dtype=np.uint8)
                vessel_only_gt[vessel_mask_gt == 1] = [255, 0, 0]
                vessel_only_gt[vessel_mask_gt == 2] = [0, 0, 255]
            else:
                vessel_only_gt = None

            # Kadran Sayımlarının Hesaplama Yapısı
            dx = mx - cx
            dy = my - cy
            if dx == 0 and dy == 0: dx = 1.0
            length = np.sqrt(dx**2 + dy**2)
            ux, uy = dx / length, dy / length
            y_indices, x_indices = np.indices((h, w))
            
            if mx - cx != 0:
                m1 = dy / dx
                c1 = cy - m1 * cx
                above_red = y_indices < (m1 * x_indices + c1)
            else:
                above_red = x_indices > cx if is_right_eye else x_indices < cx

            dot_product = (x_indices - cx) * ux + (y_indices - cy) * uy
            is_temporal = dot_product > 0

            quadrants = {}
            if is_right_eye:
                quadrants["Superior Temporal"] = above_red & is_temporal
                quadrants["Superior Nasal"]    = above_red & ~is_temporal
                quadrants["Inferior Temporal"] = ~above_red & is_temporal
                quadrants["Inferior Nasal"]    = ~above_red & ~is_temporal
            else:
                quadrants["Superior Nasal"]    = above_red & ~is_temporal
                quadrants["Superior Temporal"] = above_red & is_temporal
                quadrants["Inferior Nasal"]    = ~above_red & ~is_temporal
                quadrants["Inferior Temporal"] = ~above_red & is_temporal

            count_radius = int(od_radius * 2.5) if od_radius > 0 else 60
            ring_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(ring_mask, (int(cx), int(cy)), count_radius + 2, 255, -1)
            cv2.circle(ring_mask, (int(cx), int(cy)), count_radius - 2, 0, -1)
            ring_mask_bool = ring_mask > 0

            results = {}
            total_pred_a = 0
            total_pred_v = 0
            for q_name in ["Superior Nasal", "Superior Temporal", "Inferior Nasal", "Inferior Temporal"]:
                q_mask = quadrants[q_name]
                q_artery = ((vessel_mask_pred == 1) & q_mask & ring_mask_bool).astype(np.uint8)
                q_vein   = ((vessel_mask_pred == 2) & q_mask & ring_mask_bool).astype(np.uint8)
                
                num_arteries = int(cv2.connectedComponentsWithStats(q_artery)[0])
                num_veins    = int(cv2.connectedComponentsWithStats(q_vein)[0])
                
                final_a = (num_arteries - 1) if num_arteries > 0 else 0
                final_v = (num_veins - 1) if num_veins > 0 else 0
                
                results[q_name] = {"A": max(0, final_a), "V": max(0, final_v)}
                total_pred_a += max(0, final_a)
                total_pred_v += max(0, final_v)

            results["Total"] = {"A": total_pred_a, "V": total_pred_v}

            # Ground Truth istatistik hesaplama
            gt_results = {}
            if vessel_mask_gt is not None:
                total_gt_a = 0
                total_gt_v = 0
                for q_name in ["Superior Nasal", "Superior Temporal", "Inferior Nasal", "Inferior Temporal"]:
                    q_mask = quadrants[q_name]
                    q_artery = ((vessel_mask_gt == 1) & q_mask & ring_mask_bool).astype(np.uint8)
                    q_vein   = ((vessel_mask_gt == 2) & q_mask & ring_mask_bool).astype(np.uint8)
                    
                    num_arteries = int(cv2.connectedComponentsWithStats(q_artery)[0])
                    num_veins    = int(cv2.connectedComponentsWithStats(q_vein)[0])
                    
                    final_a = (num_arteries - 1) if num_arteries > 0 else 0
                    final_v = (num_veins - 1) if num_veins > 0 else 0
                    
                    gt_results[q_name] = {"A": max(0, final_a), "V": max(0, final_v)}
                    total_gt_a += max(0, final_a)
                    total_gt_v += max(0, final_v)
                gt_results["Total"] = {"A": total_gt_a, "V": total_gt_v}

            self.progress.emit("Analiz tamamlandı!", 100)
            
            output_data = {
                "img_name": os.path.basename(self.img_path),
                "original_img": img_rgb,
                "pred_view": final_prediction_view,
                "gt_view": final_gt_view,
                "vessel_only_pred": vessel_only_pred,
                "vessel_only_gt": vessel_only_gt,
                "od_center": od_center,
                "od_radius": od_radius,
                "macula_center": macula_center,
                "detection_type": detection_type,
                "is_right_eye": is_right_eye,
                "eye_info": eye_info,
                "results": results,
                "gt_results": gt_results if vessel_mask_gt is not None else None
            }
            self.completed.emit(output_data)

        except Exception as e:
            self.error.emit(str(e))


# --- ANA UYGULAMA PENCERESİ ---
class InfantFundusApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Sınıf değişkenleri
        self.yolo_model = None
        self.od_model = None
        self.vessel_model = None
        self.selected_file_path = None
        self.active_results = None
        
        self.setWindowTitle("👶 Bebek Fundus Analiz Sistemi - Vision AI Dashboard")
        self.setMinimumSize(1280, 850)
        
        # Modern Dark Theme Stylesheet
        self.setStyleSheet("""
            /* Genel Pencere */
            QMainWindow {
                background-color: #0b0f19;
            }
            
            /* Genel Metin Font Ayarları */
            QWidget {
                font-family: 'Segoe UI', system-ui, sans-serif;
                color: #e2e8f0;
            }
            
            /* Card & Panel */
            QFrame#ControlPanel, QFrame#ResultsFrame, QFrame#VisualsCard, QFrame#MetricsCard {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 12px;
            }
            
            /* Başlık etiketleri */
            QLabel#MainTitle {
                font-size: 20px;
                font-weight: bold;
                color: #f8fafc;
                background-color: transparent;
            }
            
            QLabel#SubTitle {
                font-size: 12px;
                color: #6366f1;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                background-color: transparent;
            }
            
            /* Butonlar */
            QPushButton {
                background-color: #6366f1;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
            QPushButton:pressed {
                background-color: #4338ca;
            }
            QPushButton:disabled {
                background-color: #374151;
                color: #9ca3af;
            }
            
            QPushButton#SecondaryButton {
                background-color: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
            }
            QPushButton#SecondaryButton:hover {
                background-color: #374151;
                border-color: #4b5563;
            }
            
            QPushButton#ExportButton {
                background-color: #059669;
            }
            QPushButton#ExportButton:hover {
                background-color: #047857;
            }
            
            /* Giriş Listesi (Sample Gallery) */
            QListWidget {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 5px;
                color: #f3f4f6;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #111827;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #374151;
                color: #ffffff;
            }
            QListWidget::item:selected {
                background-color: #6366f1;
                color: #ffffff;
                font-weight: bold;
            }
            
            /* Yazı Arama Kutusu */
            QLineEdit {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 8px 12px;
                color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #6366f1;
            }
            
            /* Drag and Drop Zone */
            QLabel#DropZone {
                border: 2px dashed #374151;
                border-radius: 10px;
                background-color: #1f2937;
                color: #9ca3af;
                font-size: 13px;
                padding: 30px 10px;
                font-weight: 500;
            }
            QLabel#DropZone[active="true"] {
                border-color: #10b981;
                background-color: #064e3b;
                color: #ecfdf5;
            }
            
            /* Tab Widget */
            QTabWidget::pane {
                border: 1px solid #1f2937;
                background-color: #111827;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #1f2937;
                color: #9ca3af;
                border: 1px solid #374151;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                margin-right: 4px;
            }
            QTabBar::tab:hover {
                background-color: #374151;
                color: #f3f4f6;
            }
            QTabBar::tab:selected {
                background-color: #111827;
                color: #ffffff;
                border-color: #1f2937;
                border-bottom: 2px solid #6366f1;
            }
            
            /* Progress Bar */
            QProgressBar {
                border: 1px solid #1f2937;
                border-radius: 6px;
                text-align: center;
                background-color: #1f2937;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #a855f7);
                border-radius: 6px;
            }
            
            /* Kadran Kartları */
            QFrame.QuadrantCard {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
            }
            QFrame.QuadrantCard:hover {
                border-color: #4f46e5;
                background-color: #242f41;
            }
            QLabel.CardTitle {
                font-weight: bold;
                font-size: 13px;
                color: #f3f4f6;
            }
            QLabel.CountValueArtery {
                font-size: 16px;
                font-weight: bold;
                color: #f87171; /* Kırmızı */
            }
            QLabel.CountValueVein {
                font-size: 16px;
                font-weight: bold;
                color: #60a5fa; /* Mavi */
            }
            QLabel.CountValueRatio {
                font-size: 15px;
                font-weight: 500;
                color: #2dd4bf; /* Yeşil-Mavi */
            }
            
            /* Splitter */
            QSplitter::handle {
                background-color: #1f2937;
                margin: 2px;
            }
            
            /* Model Yükleme Ekranı Elemanları */
            QLabel#LoadingTitle {
                font-size: 24px;
                font-weight: bold;
                color: #f8fafc;
            }
            QLabel#LoadingStatus {
                font-size: 14px;
                color: #94a3b8;
            }
        """)
        
        # Multi-page layout (StackedWidget)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 1. Sayfa: Yükleme (Splash) Ekranı
        self.setup_loading_page()
        
        # 2. Sayfa: Ana Dashboard
        self.setup_dashboard_page()
        
        # Uygulama açılır açılmaz modelleri yükleme işlemine başla
        QTimer.singleShot(200, self.start_model_loading)

    # --- 1. MODEL YÜKLEME EKRANI TASARIMI ---
    def setup_loading_page(self):
        loading_widget = QWidget()
        layout = QVBoxLayout(loading_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        # Logo veya Başlık kartı
        frame = QFrame()
        frame.setStyleSheet("background-color: #111827; border: 1px solid #1f2937; border-radius: 16px;")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(50, 40, 50, 40)
        frame_layout.setSpacing(25)
        frame_layout.setAlignment(Qt.AlignCenter)
        
        logo_label = QLabel("👁️‍🗨️")
        logo_label.setStyleSheet("font-size: 64px; background: transparent;")
        logo_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(logo_label)
        
        title = QLabel("Bebek Fundus AI Göz Analiz Sistemi")
        title.setObjectName("LoadingTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background: transparent;")
        frame_layout.addWidget(title)
        
        self.loading_status = QLabel("Yapay zeka modelleri hazırlanıyor. Lütfen bekleyin...")
        self.loading_status.setObjectName("LoadingStatus")
        self.loading_status.setAlignment(Qt.AlignCenter)
        self.loading_status.setStyleSheet("background: transparent;")
        frame_layout.addWidget(self.loading_status)
        
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0) # Sonsuz döngü (indeterminate state)
        self.loading_bar.setFixedHeight(8)
        self.loading_bar.setFixedWidth(350)
        frame_layout.addWidget(self.loading_bar)
        
        layout.addWidget(frame)
        self.stacked_widget.addWidget(loading_widget)

    # --- 2. ANA DASHBOARD EKRANI TASARIMI ---
    def setup_dashboard_page(self):
        dashboard_widget = QWidget()
        main_layout = QHBoxLayout(dashboard_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # QSplitter ile sol kontrol ve sağ görsel alanları ayır
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- SOL KONTROL PANELİ ---
        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(15)
        
        # Header (Başlık)
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)
        sub_title = QLabel("Vision AI Diagnostic")
        sub_title.setObjectName("SubTitle")
        main_title = QLabel("Fundus Analizörü")
        main_title.setObjectName("MainTitle")
        header_layout.addWidget(sub_title)
        header_layout.addWidget(main_title)
        control_layout.addLayout(header_layout)
        
        # Model Durum Göstergesi
        model_status_box = QGroupBox("🤖 Modellerin Aktiflik Durumu")
        model_status_box.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #1f2937; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        status_layout = QVBoxLayout(model_status_box)
        status_layout.setSpacing(6)
        
        self.status_yolo = QLabel("🟢 YOLOv8 Fovea/Makula: Yüklendi")
        self.status_od = QLabel("🟢 Unet++ Optik Disk: Yüklendi")
        self.status_vessel = QLabel("🟢 Unet++ Damar Segment: Yüklendi")
        
        for lbl in [self.status_yolo, self.status_od, self.status_vessel]:
            lbl.setStyleSheet("font-size: 11px; background: transparent; padding-left: 5px;")
            status_layout.addWidget(lbl)
        
        control_layout.addWidget(model_status_box)
        
        # Dosya Yükleme / Sürükleme Kartı
        upload_group = QGroupBox("📂 Fundus Görüntüsü Yükleme")
        upload_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #1f2937; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(10)
        
        self.drop_zone = DropZoneLabel()
        self.drop_zone.fileDropped.connect(self.load_image_from_path)
        upload_layout.addWidget(self.drop_zone)
        
        btn_select_file = QPushButton("📁 Dosya Gezgininden Seç")
        btn_select_file.setObjectName("SecondaryButton")
        btn_select_file.clicked.connect(self.select_file)
        upload_layout.addWidget(btn_select_file)
        
        control_layout.addWidget(upload_group)
        
        # Test Dosyaları Listesi (Sample Gallery)
        samples_group = QGroupBox("📊 Test Veri Seti Galerisi")
        samples_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #1f2937; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        samples_layout = QVBoxLayout(samples_group)
        samples_layout.setSpacing(8)
        
        # Search Box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Görüntü adı ara...")
        self.search_box.textChanged.connect(self.filter_samples)
        samples_layout.addWidget(self.search_box)
        
        # QListWidget
        self.samples_list = QListWidget()
        self.samples_list.itemDoubleClicked.connect(self.load_sample_item)
        samples_layout.addWidget(self.samples_list)
        
        control_layout.addWidget(samples_group)
        
        # Analizi Başlat Butonu
        self.btn_analyze = QPushButton("🚀 Analizi Başlat")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.run_fundus_analysis)
        control_layout.addWidget(self.btn_analyze)
        
        # Progress Bar ve Analiz Durumu
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setVisible(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setFixedHeight(12)
        control_layout.addWidget(self.analysis_progress_bar)
        
        self.analysis_status_label = QLabel("")
        self.analysis_status_label.setStyleSheet("font-size: 11px; color: #a855f7; font-weight: 500;")
        self.analysis_status_label.setAlignment(Qt.AlignCenter)
        self.analysis_status_label.setVisible(False)
        control_layout.addWidget(self.analysis_status_label)
        
        # Sol paneli splitter'a ekle ve varsayılan genişlik ayarla
        splitter.addWidget(control_panel)
        
        # --- SAĞ İÇERİK/SONUÇ ALANI ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        
        # Görsel Sonuç Tabları
        visuals_card = QFrame()
        visuals_card.setObjectName("VisualsCard")
        visuals_layout = QVBoxLayout(visuals_card)
        visuals_layout.setContentsMargins(10, 10, 10, 10)
        
        self.results_tabs = QTabWidget()
        visuals_layout.addWidget(self.results_tabs)
        
        # Tab 1: Hibrit Analiz Görünümü (Yan Yana)
        tab_hybrid = QWidget()
        hybrid_layout = QHBoxLayout(tab_hybrid)
        hybrid_layout.setContentsMargins(5, 10, 5, 5)
        hybrid_layout.setSpacing(10)
        
        # Orijinal Görüntü Kartı
        orig_container = QGroupBox("Orijinal Giriş Görüntüsü")
        orig_container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        orig_lay = QVBoxLayout(orig_container)
        self.img_label_orig = ScaledImageLabel()
        self.img_label_orig.setText("Buraya analiz için bir resim yükleyin.")
        self.img_label_orig.setStyleSheet("color: #6b7280; font-size: 13px;")
        orig_lay.addWidget(self.img_label_orig)
        hybrid_layout.addWidget(orig_container)
        
        # Tahmin Sonucu Görüntü Kartı
        pred_container = QGroupBox("Model Analiz Sonucu")
        pred_container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        pred_lay = QVBoxLayout(pred_container)
        self.img_label_pred = ScaledImageLabel()
        self.img_label_pred.setText("Analiz başlatıldıktan sonra sonuç burada görüntülenecektir.")
        self.img_label_pred.setStyleSheet("color: #6b7280; font-size: 13px;")
        pred_lay.addWidget(self.img_label_pred)
        hybrid_layout.addWidget(pred_container)
        
        self.results_tabs.addTab(tab_hybrid, "🔬 Kadran & Segmentasyon Analizi")
        
        # Tab 2: Sadece Damar Maskeleri (Yan Yana)
        tab_vessels = QWidget()
        vessels_layout = QHBoxLayout(tab_vessels)
        vessels_layout.setContentsMargins(5, 10, 5, 5)
        vessels_layout.setSpacing(10)
        
        v_orig_container = QGroupBox("Giriş Görüntüsü")
        v_orig_container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        v_orig_lay = QVBoxLayout(v_orig_container)
        self.img_label_v_orig = ScaledImageLabel()
        v_orig_lay.addWidget(self.img_label_v_orig)
        vessels_layout.addWidget(v_orig_container)
        
        v_pred_container = QGroupBox("Damar Segmentasyon Maskesi (Mavi: Ven, Kırmızı: Arter)")
        v_pred_container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        v_pred_lay = QVBoxLayout(v_pred_container)
        self.img_label_v_pred = ScaledImageLabel()
        v_pred_lay.addWidget(self.img_label_v_pred)
        vessels_layout.addWidget(v_pred_container)
        
        self.results_tabs.addTab(tab_vessels, "🩸 İzole Damar Yatağı Maskesi")

        # Tab 3: Ground Truth Karşılaştırma Görünümü (Eğer varsa)
        self.tab_gt = QWidget()
        gt_layout = QHBoxLayout(self.tab_gt)
        gt_layout.setContentsMargins(5, 10, 5, 5)
        gt_layout.setSpacing(10)
        
        gt_view_container = QGroupBox("Ground Truth (Referans Etiket)")
        gt_view_container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        gt_view_lay = QVBoxLayout(gt_view_container)
        self.img_label_gt = ScaledImageLabel()
        gt_view_lay.addWidget(self.img_label_gt)
        gt_layout.addWidget(gt_view_container)
        
        gt_pred_container = QGroupBox("Yapay Zeka Tahmini")
        gt_pred_container.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        gt_pred_lay = QVBoxLayout(gt_pred_container)
        self.img_label_gt_pred = ScaledImageLabel()
        gt_pred_lay.addWidget(self.img_label_gt_pred)
        gt_layout.addWidget(gt_pred_container)
        
        # Başlangıçta GT tabını ekleme (aktif bir sonuç olduğunda ve GT mevcutsa ekleyeceğiz)
        
        right_layout.addWidget(visuals_card, stretch=4)
        
        # --- METRİKLER VE SAYIM DASHBOARDU ---
        metrics_card = QFrame()
        metrics_card.setObjectName("MetricsCard")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(15, 12, 15, 12)
        metrics_layout.setSpacing(10)
        
        # Üst Satır: Göz Yönü, Tespit Metodu ve Rapor Çıktı Butonları
        meta_layout = QHBoxLayout()
        
        self.lbl_eye_info = QLabel("👁️ Göz Yönü: Tespit Edilmedi")
        self.lbl_eye_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #a855f7;")
        meta_layout.addWidget(self.lbl_eye_info)
        
        self.lbl_method_info = QLabel("🛠️ Fovea Metodu: Tespit Edilmedi")
        self.lbl_method_info.setStyleSheet("font-size: 13px; color: #94a3b8; font-weight: 500;")
        meta_layout.addWidget(self.lbl_method_info)
        
        meta_layout.addStretch()
        
        self.btn_export = QPushButton("📥 Klinik Raporu Kaydet")
        self.btn_export.setObjectName("ExportButton")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_clinical_report)
        meta_layout.addWidget(self.btn_export)
        
        metrics_layout.addLayout(meta_layout)
        
        # Orta Kısım: Kadran İstatistikleri (2x2 Grid)
        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)
        
        # Kadran Kartlarını Oluştur
        self.quadrant_cards = {}
        quadrant_names = [
            ("Superior Temporal (ST)", 0, 0),
            ("Superior Nasal (SN)", 0, 1),
            ("Inferior Temporal (IT)", 1, 0),
            ("Inferior Nasal (IN)", 1, 1)
        ]
        
        for name, row, col in quadrant_names:
            card = QFrame()
            card.setObjectName("QuadrantCard")
            card.setProperty("class", "QuadrantCard")
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(10, 10, 10, 10)
            card_lay.setSpacing(4)
            
            lbl_title = QLabel(name)
            lbl_title.setObjectName("CardTitle")
            lbl_title.setProperty("class", "CardTitle")
            card_lay.addWidget(lbl_title)
            
            counts_lay = QHBoxLayout()
            lbl_art = QLabel("Arter (A): 0")
            lbl_art.setObjectName("CountValueArtery")
            lbl_art.setProperty("class", "CountValueArtery")
            lbl_vein = QLabel("Ven (V): 0")
            lbl_vein.setObjectName("CountValueVein")
            lbl_vein.setProperty("class", "CountValueVein")
            
            counts_lay.addWidget(lbl_art)
            counts_lay.addWidget(lbl_vein)
            card_lay.addLayout(counts_lay)
            
            lbl_ratio = QLabel("A/V Oranı: 0.00")
            lbl_ratio.setObjectName("CountValueRatio")
            lbl_ratio.setProperty("class", "CountValueRatio")
            card_lay.addWidget(lbl_ratio)
            
            grid_layout.addWidget(card, row, col)
            
            # Kart referansını sakla
            self.quadrant_cards[name.split(" (")[0]] = {
                "frame": card,
                "lbl_art": lbl_art,
                "lbl_vein": lbl_vein,
                "lbl_ratio": lbl_ratio
            }
            
        # Toplam Sayımları Gösteren Kart (Grid'in sağına veya altına yerleşecek)
        total_card = QFrame()
        total_card.setStyleSheet("background-color: #1e1b4b; border: 1px solid #312e81; border-radius: 8px;")
        total_lay = QVBoxLayout(total_card)
        total_lay.setContentsMargins(12, 10, 12, 10)
        total_lay.setSpacing(4)
        
        lbl_tot_title = QLabel("Tüm Göz Geneli")
        lbl_tot_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #a5b4fc;")
        total_lay.addWidget(lbl_tot_title)
        
        tot_counts_lay = QHBoxLayout()
        self.lbl_tot_art = QLabel("Toplam Arter: 0")
        self.lbl_tot_art.setStyleSheet("font-size: 14px; font-weight: bold; color: #fca5a5;")
        self.lbl_tot_vein = QLabel("Toplam Ven: 0")
        self.lbl_tot_vein.setStyleSheet("font-size: 14px; font-weight: bold; color: #93c5fd;")
        tot_counts_lay.addWidget(self.lbl_tot_art)
        tot_counts_lay.addWidget(self.lbl_tot_vein)
        total_lay.addLayout(tot_counts_lay)
        
        self.lbl_tot_ratio = QLabel("Global A/V Oranı: 0.00")
        self.lbl_tot_ratio.setStyleSheet("font-size: 14px; font-weight: bold; color: #2dd4bf;")
        total_lay.addWidget(self.lbl_tot_ratio)
        
        # Kadran grid'inin yanına toplam kartını ekle
        bottom_grid_layout = QHBoxLayout()
        bottom_grid_layout.addWidget(grid_container, stretch=3)
        bottom_grid_layout.addWidget(total_card, stretch=1)
        
        metrics_layout.addLayout(bottom_grid_layout)
        
        right_layout.addWidget(metrics_card, stretch=1)
        
        splitter.addWidget(right_panel)
        
        # Splitter oranlarını ayarla (Sol: 300px, Sağ: Kalan alan)
        splitter.setSizes([320, 960])
        
        self.stacked_widget.addWidget(dashboard_widget)

    # --- 3. MODEL YÜKLEME SÜRECİNİ BAŞLATMA ---
    def start_model_loading(self):
        self.loader_thread = ModelLoaderWorker()
        self.loader_thread.status.connect(self.update_loading_status)
        self.loader_thread.loaded.connect(self.on_models_loaded)
        self.loader_thread.error.connect(self.on_models_load_error)
        self.loader_thread.start()

    def update_loading_status(self, text):
        self.loading_status.setText(text)

    def on_models_loaded(self, yolo_model, od_model, vessel_model):
        self.yolo_model = yolo_model
        self.od_model = od_model
        self.vessel_model = vessel_model
        
        # Ana ekrana geçiş yap
        self.stacked_widget.setCurrentIndex(1)
        
        # Test klasöründeki dosyaları listele
        self.load_samples_to_list()

    def on_models_load_error(self, error_str):
        self.loading_bar.setVisible(False)
        self.loading_status.setText(f"❌ Modeller yüklenirken bir hata oluştu:\n{error_str}")
        self.loading_status.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")
        QMessageBox.critical(
            self, 
            "Model Yükleme Hatası", 
            f"Derin öğrenme modelleri yüklenemedi. Lütfen model yollarının doğruluğunu kontrol edin.\n\nHata: {error_str}"
        )

    # --- 4. TEST DOSYALARINI LİSTELEME VE FİLTRELEME ---
    def load_samples_to_list(self):
        if not os.path.exists(test_img_dir):
            return
        
        self.samples_list.clear()
        img_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Sırala ve listeye ekle
        for filename in sorted(img_files):
            item = QListWidgetItem(filename)
            # İsteğe bağlı: maske dosyasının var olup olmadığını kontrol edip yanına bir ikon koyabiliriz
            mask_exists = False
            base_name = os.path.splitext(filename)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.TIF']:
                if os.path.exists(os.path.join(test_mask_dir, base_name + ext)):
                    mask_exists = True
                    break
            
            if mask_exists:
                item.setText(f"🏷️ {filename} [GT Var]")
            else:
                item.setText(f"🖼️ {filename}")
                
            # Orijinal dosya adını veri (data) olarak ata
            item.setData(Qt.UserRole, filename)
            self.samples_list.addItem(item)

    def filter_samples(self, text):
        for i in range(self.samples_list.count()):
            item = self.samples_list.item(i)
            file_name = item.data(Qt.UserRole)
            item.setHidden(text.lower() not in file_name.lower())

    # --- 5. GÖRÜNTÜ YÜKLEME METOTLARI ---
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Fundus Görüntüsü Seç",
            test_img_dir if os.path.exists(test_img_dir) else "",
            "Görüntü Dosyaları (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.load_image_from_path(file_path)

    def load_sample_item(self, item):
        filename = item.data(Qt.UserRole)
        file_path = os.path.join(test_img_dir, filename)
        self.load_image_from_path(file_path)

    def load_image_from_path(self, file_path):
        self.selected_file_path = file_path
        
        # Orijinal görüntüyü yükle ve sol kutuya çiz
        img_bgr = cv2.imread(file_path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pixmap = cv2_to_qpixmap(img_rgb)
            
            self.img_label_orig.setPixmap(pixmap)
            self.img_label_v_orig.setPixmap(pixmap)
            
            self.drop_zone.setText(f"Seçilen Dosya:\n📄 {os.path.basename(file_path)}")
            self.drop_zone.setStyleSheet("""
                QLabel#DropZone {
                    border: 2px solid #6366f1;
                    background-color: #1e293b;
                    color: #e5e7eb;
                    font-size: 12px;
                    padding: 20px 10px;
                }
            """)
            
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setText("🚀 Analizi Başlat")
            self.btn_analyze.setStyleSheet("background-color: #6366f1; color: white;")
            
            # Eski sonuçları sıfırla
            self.img_label_pred.setText("Analiz başlatıldıktan sonra sonuç burada görüntülenecektir.")
            self.img_label_pred.setStyleSheet("color: #6b7280; font-size: 13px;")
            self.img_label_v_pred.clear()
            self.img_label_gt.clear()
            self.img_label_gt_pred.clear()
            
            # Ground truth tabını kaldır (varsa sıfırlansın)
            if self.results_tabs.count() > 2:
                self.results_tabs.removeTab(2)
                
            self.btn_export.setEnabled(False)
            self.active_results = None
            self.reset_metrics_dashboard()

    def reset_metrics_dashboard(self):
        self.lbl_eye_info.setText("👁️ Göz Yönü: Tespit Edilmedi")
        self.lbl_method_info.setText("🛠️ Fovea Metodu: Tespit Edilmedi")
        self.lbl_tot_art.setText("Toplam Arter: 0")
        self.lbl_tot_vein.setText("Toplam Ven: 0")
        self.lbl_tot_ratio.setText("Global A/V Oranı: 0.00")
        
        for q_name, card in self.quadrant_cards.items():
            card["lbl_art"].setText("Arter (A): 0")
            card["lbl_vein"].setText("Ven (V): 0")
            card["lbl_ratio"].setText("A/V Oranı: 0.00")
            # Normal stilde sıfırla
            card["frame"].setStyleSheet("""
                QFrame.QuadrantCard {
                    background-color: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 8px;
                }
            """)

    # --- 6. HİBRİT ANALİZ EXECUTION SÜRECİ ---
    def run_fundus_analysis(self):
        if not self.selected_file_path:
            return
        
        # Butonu ve kontrolleri pasif et, progress barı aç
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText("🔄 Analiz Ediliyor...")
        self.btn_analyze.setStyleSheet("background-color: #4b5563; color: #9ca3af;")
        self.samples_list.setEnabled(False)
        
        self.analysis_progress_bar.setVisible(True)
        self.analysis_progress_bar.setValue(0)
        self.analysis_status_label.setVisible(True)
        self.analysis_status_label.setText("İşlem başlatılıyor...")
        
        # Worker thread'i kur
        self.worker = AnalysisWorker(
            self.selected_file_path,
            self.yolo_model,
            self.od_model,
            self.vessel_model
        )
        self.worker.progress.connect(self.on_analysis_progress)
        self.worker.completed.connect(self.on_analysis_completed)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_progress(self, message, percent):
        self.analysis_progress_bar.setValue(percent)
        self.analysis_status_label.setText(message)

    def on_analysis_completed(self, results):
        self.active_results = results
        
        # Arayüz kontrollerini tekrar aktifleştir
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setText("🚀 Analizi Tekrar Başlat")
        self.btn_analyze.setStyleSheet("background-color: #6366f1; color: white;")
        self.samples_list.setEnabled(True)
        
        self.analysis_progress_bar.setVisible(False)
        self.analysis_status_label.setVisible(False)
        
        # 1. Görüntüleri Tablara Yükle
        pix_pred = cv2_to_qpixmap(results["pred_view"])
        self.img_label_pred.setPixmap(pix_pred)
        
        pix_vessel_only_pred = cv2_to_qpixmap(results["vessel_only_pred"])
        self.img_label_v_pred.setPixmap(pix_vessel_only_pred)
        
        # Ground Truth varsa Tab 3'ü ekle ve yükle
        if results["gt_view"] is not None:
            # Önce temizle (varsa eski tab kalmasın)
            if self.results_tabs.count() > 2:
                self.results_tabs.removeTab(2)
                
            self.results_tabs.addTab(self.tab_gt, "🔬 Referans (Ground Truth) Karşılaştırma")
            
            pix_gt = cv2_to_qpixmap(results["gt_view"])
            self.img_label_gt.setPixmap(pix_gt)
            self.img_label_gt_pred.setPixmap(pix_pred)
        
        # 2. Metrikleri Güncelle
        self.lbl_eye_info.setText(f"👁️ Göz Anatomisi: {results['eye_info']}")
        self.lbl_method_info.setText(f"🎯 Makula Metodu: {results['detection_type']}")
        
        # Kadran Kartlarını ve Renklendirmeleri Doldur
        q_results = results["results"]
        for q_name, counts in q_results.items():
            if q_name == "Total":
                continue
            
            card = self.quadrant_cards.get(q_name)
            if card:
                a_cnt = counts["A"]
                v_cnt = counts["V"]
                ratio = a_cnt / v_cnt if v_cnt > 0 else (a_cnt if a_cnt > 0 else 0.0)
                
                card["lbl_art"].setText(f"Arter (A): {a_cnt}")
                card["lbl_vein"].setText(f"Ven (V): {v_cnt}")
                
                ratio_str = f"{ratio:.2f}" if v_cnt > 0 else "N/A"
                card["lbl_ratio"].setText(f"A/V Oranı: {ratio_str}")
                
                # Klinik Uyarı: ROP veya anormal damar durumlarında A/V oranı genelde düşer (arter daralır/ven genişler)
                # Normal sınırları kabaca [0.55 - 0.85] alalım
                if v_cnt > 0 and (ratio < 0.50 or ratio > 0.95):
                    # Kırmızı/Turuncu Uyarı Çerçevesi
                    card["frame"].setStyleSheet("""
                        QFrame.QuadrantCard {
                            background-color: #2d1a1a;
                            border: 1px solid #ef4444;
                            border-radius: 8px;
                        }
                    """)
                else:
                    # Normal Yeşil/Mavi Çerçeve
                    card["frame"].setStyleSheet("""
                        QFrame.QuadrantCard {
                            background-color: #1a2e26;
                            border: 1px solid #10b981;
                            border-radius: 8px;
                        }
                    """)
        
        # Toplam Kartı Güncelle
        tot_a = q_results["Total"]["A"]
        tot_v = q_results["Total"]["V"]
        tot_ratio = tot_a / tot_v if tot_v > 0 else (tot_a if tot_a > 0 else 0.0)
        
        self.lbl_tot_art.setText(f"Toplam Arter: {tot_a}")
        self.lbl_tot_vein.setText(f"Toplam Ven: {tot_v}")
        
        tot_ratio_str = f"{tot_ratio:.2f}" if tot_v > 0 else "N/A"
        self.lbl_tot_ratio.setText(f"Global A/V Oranı: {tot_ratio_str}")
        
        self.btn_export.setEnabled(True)
        QMessageBox.information(
            self, 
            "Analiz Tamamlandı", 
            f"Fundus analizi tamamlandı!\nTespit edilen anatomi: {results['eye_info']}\nToplam Damar Sayımı: Arter {tot_a} | Ven {tot_v}"
        )

    def on_analysis_error(self, error_str):
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setText("🚀 Analizi Başlat")
        self.btn_analyze.setStyleSheet("background-color: #6366f1; color: white;")
        self.samples_list.setEnabled(True)
        
        self.analysis_progress_bar.setVisible(False)
        self.analysis_status_label.setVisible(False)
        
        QMessageBox.critical(
            self, 
            "Analiz Hatası", 
            f"Görüntü analiz edilirken bir hata oluştu.\n\nHata: {error_str}"
        )

    # --- 7. RAPOR DIŞA AKTARMA ---
    def export_clinical_report(self):
        if not self.active_results:
            return
        
        # Rapor dosyasının adını kaydetmek için kaydetme penceresi aç
        default_report_name = f"klinik_rapor_{os.path.splitext(self.active_results['img_name'])[0]}.txt"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Klinik Raporu Kaydet",
            os.path.join(os.path.expanduser("~"), "Desktop", default_report_name),
            "Metin Belgeleri (*.txt)"
        )
        
        if not save_path:
            return
            
        try:
            results = self.active_results
            q_results = results["results"]
            
            report_content = []
            report_content.append("="*60)
            report_content.append("          BEBEK FUNDUS VİZYON AI ANALİZ SİSTEMİ KLİNİK RAPORU")
            report_content.append("="*60)
            report_content.append(f"Görüntü Dosyası   : {results['img_name']}")
            report_content.append(f"Tespit Edilen Göz : {results['eye_info']}")
            report_content.append(f"Makula Tespit Yöntemi: {results['detection_type']}")
            report_content.append(f"Optik Disk Merkezi: X: {results['od_center'][0]}, Y: {results['od_center'][1]} | Yarıçap: {results['od_radius']} px")
            report_content.append(f"Makula Merkezi    : X: {results['macula_center'][0]}, Y: {results['macula_center'][1]}")
            report_content.append("-"*60)
            report_content.append("                      DAMAR VE KADRAN İSTATİSTİKLERİ")
            report_content.append("-"*60)
            
            # Kadranları yazdır
            for q_name in ["Superior Temporal", "Superior Nasal", "Inferior Temporal", "Inferior Nasal"]:
                counts = q_results[q_name]
                a = counts["A"]
                v = counts["V"]
                ratio = a / v if v > 0 else (a if a > 0 else 0.0)
                ratio_str = f"{ratio:.2f}" if v > 0 else "N/A (Sıfır Ven)"
                
                report_content.append(f"• {q_name:<20}: Arter: {a:<4} | Ven: {v:<4} | A/V Oranı: {ratio_str}")
                
            report_content.append("-"*60)
            tot_a = q_results["Total"]["A"]
            tot_v = q_results["Total"]["V"]
            tot_ratio = tot_a / tot_v if tot_v > 0 else (tot_a if tot_a > 0 else 0.0)
            tot_ratio_str = f"{tot_ratio:.2f}" if tot_v > 0 else "N/A"
            report_content.append(f"GÖZ GENELİ TOPLAM   : Toplam Arter: {tot_a:<4} | Toplam Ven: {tot_v:<4} | Global A/V: {tot_ratio_str}")
            
            # Ground truth varsa onu da rapora ekle
            if results["gt_results"] is not None:
                gt_q_results = results["gt_results"]
                report_content.append("="*60)
                report_content.append("          REFERANS (GROUND TRUTH) ETİKET DEĞERLERİ KARŞILAŞTIRMA")
                report_content.append("="*60)
                for q_name in ["Superior Temporal", "Superior Nasal", "Inferior Temporal", "Inferior Nasal"]:
                    counts_gt = gt_q_results[q_name]
                    a_gt = counts_gt["A"]
                    v_gt = counts_gt["V"]
                    ratio_gt = a_gt / v_gt if v_gt > 0 else (a_gt if a_gt > 0 else 0.0)
                    ratio_gt_str = f"{ratio_gt:.2f}" if v_gt > 0 else "N/A"
                    
                    report_content.append(f"• {q_name:<20}: Arter: {a_gt:<4} | Ven: {v_gt:<4} | A/V Oranı: {ratio_gt_str}")
                    
                tot_gt_a = gt_q_results["Total"]["A"]
                tot_gt_v = gt_q_results["Total"]["V"]
                tot_gt_ratio = tot_gt_a / tot_gt_v if tot_gt_v > 0 else (tot_gt_a if tot_gt_a > 0 else 0.0)
                tot_gt_ratio_str = f"{tot_gt_ratio:.2f}" if tot_gt_v > 0 else "N/A"
                report_content.append(f"REFERANS TOPLAM     : Toplam Arter: {tot_gt_a:<4} | Toplam Ven: {tot_gt_v:<4} | Global A/V: {tot_gt_ratio_str}")
            
            report_content.append("\n* Bu rapor Bebek Fundus AI Göz Analiz Sistemi tarafından otomatik üretilmiştir.")
            report_content.append("*" * 60)
            
            # Dosyaya kaydet
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_content))
                
            # Ayrıca görselleri kaydet
            base_img_path = os.path.splitext(save_path)[0]
            cv2.imwrite(f"{base_img_path}_analiz.png", cv2.cvtColor(results["pred_view"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{base_img_path}_damar_maske.png", cv2.cvtColor(results["vessel_only_pred"], cv2.COLOR_RGB2BGR))
            
            QMessageBox.information(
                self, 
                "Başarılı", 
                f"Klinik rapor metni ve analiz görselleri başarıyla kaydedildi!\n\nMetin: {os.path.basename(save_path)}\nResimler: {os.path.basename(base_img_path)}_analiz.png"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Kayıt Hatası", 
                f"Rapor dosyaya kaydedilirken bir hata oluştu.\n\nHata: {e}"
            )


# --- ÇALIŞTIRICI ---
if __name__ == "__main__":
    # DPI ölçeklendirmesini aktifleştir (High DPI Windows ekranları için net görüntü)
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    
    # Modern palet ayarla
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0b0f19"))
    palette.setColor(QPalette.WindowText, QColor("#e2e8f0"))
    palette.setColor(QPalette.Base, QColor("#1f2937"))
    palette.setColor(QPalette.AlternateBase, QColor("#111827"))
    palette.setColor(QPalette.ToolTipBase, QColor("#f8fafc"))
    palette.setColor(QPalette.ToolTipText, QColor("#0f172a"))
    palette.setColor(QPalette.Text, QColor("#ffffff"))
    palette.setColor(QPalette.Button, QColor("#6366f1"))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#6366f1"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    
    window = InfantFundusApp()
    window.show()
    sys.exit(app.exec_())
