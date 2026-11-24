"""Local SSD300 inference shared by the marimo notebook and smoke checks."""

from io import BytesIO
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torchvision import transforms
from torchvision.ops import batched_nms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import gcxgcy_to_cxcy, cxcy_to_xy, label_color_map, rev_label_map


def read_image(contents: bytes) -> Image.Image:
    """Decode uploads and camera snapshots, including phone EXIF orientation."""
    with Image.open(BytesIO(contents)) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


class Detector:
    def __init__(self, checkpoint_path, device='auto'):
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'No existe el checkpoint: {path}')
        if device == 'auto':
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device(device)
        # These locally trained checkpoints serialize the model and optimizer.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model = checkpoint['model'].to(self.device).eval()
        # Priors in this tutorial are ordinary tensors, not registered buffers.
        self.priors = self.model.priors_cxcy.detach().cpu()
        self.epoch = checkpoint.get('epoch', -1) + 1
        self.steps = checkpoint.get('steps')
        self.path = path.resolve()
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict(self, image, min_score=0.4, max_overlap=0.45, top_k=50):
        """Forward on the selected device; decode and class-wise NMS on CPU."""
        start = perf_counter()
        image = image.convert('RGB')
        inputs = self.transform(image).unsqueeze(0).to(self.device)
        locations, logits = self.model(inputs)
        locations, logits = locations[0].cpu(), logits[0].cpu()
        boxes = cxcy_to_xy(gcxgcy_to_cxcy(locations, self.priors))
        scores = logits.softmax(dim=-1)
        # A prior may qualify for several classes, as in SSD300.detect_objects.
        candidates = scores[:, 1:] > min_score
        candidates &= torch.isfinite(boxes).all(dim=1).unsqueeze(1)
        candidates &= (boxes[:, 2:] > boxes[:, :2]).all(dim=1).unsqueeze(1)
        prior_ids, classes = candidates.nonzero(as_tuple=True)
        classes = classes + 1
        boxes = boxes[prior_ids]
        scores = scores[prior_ids, classes]
        # CPU NMS also avoids dependence on MPS support for torchvision NMS.
        keep = batched_nms(boxes, scores, classes, max_overlap)[:top_k]
        boxes, scores, classes = boxes[keep].clamp(0, 1), scores[keep], classes[keep]
        boxes *= torch.tensor([image.width, image.height, image.width, image.height])

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        rows = []
        for box, score, class_id in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            label = rev_label_map[class_id]
            color = label_color_map[label]
            x1, y1, x2, y2 = box
            draw.rectangle(box, outline=color, width=3)
            text = f'{label} {score:.0%}'
            bounds = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            text_x = max(0, min(x1, image.width - text_width - 6))
            text_y = max(0, y1 - text_height - 6)
            draw.rectangle((text_x, text_y, text_x + text_width + 6, text_y + text_height + 6), fill=color)
            draw.text((text_x + 3, text_y + 3 - bounds[1]), text, font=font, fill='white')
            rows.append({'Clase': label, 'Confianza (%)': round(score * 100, 1),
                         'x1': round(x1), 'y1': round(y1), 'x2': round(x2), 'y2': round(y2)})
        return annotated, rows, perf_counter() - start
