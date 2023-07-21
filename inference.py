from model8 import Lipreading
from get_checkpoint_update import get_checkpoint_dir
import torch
import cv2
from torchvision import transforms
from PIL import Image
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

CKPT_PATH = "ckpt/custom_model8_3.pt"

def make_input_tensor(video_path):

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.522], std=[0.125]),
        ])
    
    input_tensor_list = []
    video_name = video_path.split("/")[-1]
    
    cap = cv2.VideoCapture(video_path)
    total_frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
                break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_pil = Image.fromarray(frame)
        input_tensor = preprocess(frame_pil)
        input_tensor_list.append(input_tensor)
    
    input_sequence = torch.stack(input_tensor_list)

    zeros_tensor = torch.zeros((550, 1, 96, 96))

    # 원본 입력 텐서를 0으로 채운 텐서와 결합하여 크기를 확장
    input_sequence = torch.cat([input_sequence, zeros_tensor[total_frame_ct:, :, :, :]], dim=0)
    del zeros_tensor
    torch.cuda.empty_cache()
    
    return input_sequence, video_name, total_frame_ct

video_path = "bf2c632e-6641-4ff7-a905-7f8d9a64ea80_preprocessed.mp4"
get_checkpoint_dir()

with torch.no_grad():
    model = Lipreading()

    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict["model_cnn"])

    model = model.to(device)
    model.eval()

    input_sequence, video_name, total_frame_ct = make_input_tensor(video_path)
    input_sequence = input_sequence.unsqueeze(0)
    input_sequence = input_sequence.transpose(1, 2)
    input_sequence = input_sequence.to(device)

    y_hat = model(input_sequence)
    y_hat = y_hat.squeeze(0)
    print(y_hat)