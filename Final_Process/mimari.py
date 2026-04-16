import segmentation_models_pytorch as smp
import torch.nn as nn

def get_segmentation_model(num_classes=1, encoder_name="efficientnet-b3"):
    """
    En iyi sonuç için hibrit mimari:
    - Unet++: İç içe geçmiş skip connectionlar ile daha zengin özellik aktarımı.
    - scSE Attention: Önemli piksellere ve kanallara odaklanma.
    - EfficientNet-B3: Dengeli derinlik ve genişlikte güçlü bir backbone.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,         # EfficientNet-B3
        encoder_weights="imagenet",       # ImageNet ön eğitimi ile başlar
        in_channels=3,                    # RGB giriş
        classes=num_classes,              # Çıktı (Maske)
        activation='sigmoid',             # Binary segmentasyon için (0-1 arası)
        decoder_attention_type='scse'     # Spatial & Channel Attention mekanizması
    )
    
    return model

class HybridLoss(nn.Module):
    """
    Daha keskin sınırlar için Dice ve BCE kaybını birleştiren özel class.
    Dice Loss: Genel forma (IoU) odaklanır.
    BCE Loss: Piksel bazlı doğruluğa odaklanır.
    """
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = smp.losses.SoftBCEWithLogitsLoss() # Logitler üzerinde daha kararlı çalışır

    def forward(self, y_pred, y_true):
        # Eğer modelde sigmoid varsa, SoftBCE yerine normal BCE kullanılabilir
        # Ancak SMP modelleri genellikle kararlılık için ham logitlerle daha iyi çalışır.
        # Burada basitleştirmek için direkt Dice + BCE mantığını kuruyoruz:
        return 0.5 * self.dice(y_pred, y_true) + 0.5 * nn.BCELoss()(y_pred, y_true)

def get_loss_function():
    return HybridLoss()