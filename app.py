import os

import cv2
import gradio as gr
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

os.system("pip freeze")
# download weights
if not os.path.exists('realesr-general-x4v3.pth'):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
if not os.path.exists('GFPGANv1.2.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P .")
if not os.path.exists('GFPGANv1.3.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .")
if not os.path.exists('GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('RestoreFormer.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P .")


# background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

os.makedirs('output', exist_ok=True)


# def inference(img, version, scale, weight):
def inference(img, version, scale):
    # weight /= 100
    print(img, version, scale)
    if scale > 4:
        scale = 4  # avoid too large scale value
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            print('too large size')
            return None, None

        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'v1.2':
            face_enhancer = GFPGANer(
            model_path='GFPGANv1.2.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.3':
            face_enhancer = GFPGANer(
            model_path='GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.4':
            face_enhancer = GFPGANer(
            model_path='GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'RestoreFormer':
            face_enhancer = GFPGANer(
            model_path='RestoreFormer.pth', upscale=2, arch='RestoreFormer', channel_multiplier=2, bg_upsampler=upsampler)
        # elif version == 'CodeFormer':
        #     face_enhancer = GFPGANer(
        #     model_path='CodeFormer.pth', upscale=2, arch='CodeFormer', channel_multiplier=2, bg_upsampler=upsampler)

        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output/out.{extension}'
        cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('global exception', error)
        return None, None


title = "DIAMONIK7777- GFPGAN: Practical Face Restoration Algorithm"
description = "<p style='text-align: center'>Будь в курсе обновлений <a href='https://vk.com/public221489796'>ПОДПИСАТЬСЯ</a></p>"
article = "<br><br><p style='text-align: center'>Генерация индивидуальной модели с собственной внешностью <a href='https://vk.com/im?sel=-221489796'>ПОДАТЬ ЗАЯВКУ</a></p><br><br><br><br><br>"
gfpgan = gr.Interface(
    inference, [
        gr.Image(type="filepath", label="Input"),
        gr.Radio(['v1.2', 'v1.3', 'v1.4', 'RestoreFormer'], type="value", value='v1.4', label='Версия'),
        gr.Number(label="Коэффициент масштабирования", value=2),
        # gr.Slider(0, 100, label='Weight, only for CodeFormer. 0 for better quality, 100 for better identity', value=50)
    ], [
        gr.Image(type="numpy", label="Output (The whole image)"),
        gr.File(label="Download the output image")
    ],
    title=title,
    description=description,
    article=article,
     )
gfpgan.queue().launch(debug=True, max_threads=True, share=True, inbrowser=True)
