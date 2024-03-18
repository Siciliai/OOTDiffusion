import base64
import io
from pathlib import Path
import sys

import torch
from PIL import Image
import numpy as np

project_dir = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(project_dir))

from utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
try:
    from preprocess.humanparsing.aigc_run_parsing import Parsing
except ModuleNotFoundError:
    from preprocess.humanparsing.run_parsing import Parsing

from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


openpose_model = OpenPose(0)
parsing_model = Parsing(0)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = "hd"  # "hd" or "dc"
category = 0  # 0:upperbody; 1:lowerbody; 2:dress
image_scale = 2.0
n_steps = 20
n_samples = 1
seed = 1

if model_type == "hd":
    model = OOTDiffusionHD(0)
elif model_type == "dc":
    model = OOTDiffusionDC(0)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")


class OOTDGenError(Exception):
    pass


def gen_trying_image(model_data, cloth_data, mask_data):
    """
    生成试穿图片
    """
    cloth_img = Image.open(io.BytesIO(base64.b64decode(cloth_data))).resize((768, 1024))
    cloth_img = cloth_img.convert('RGB')

    model_img = Image.open(io.BytesIO(base64.b64decode(model_data))).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_data)))
    mask = mask_img.convert('L')
    array = np.array(mask)
    mask_gray_array = (array / 2).astype(np.uint8)
    mask_gray = Image.fromarray(mask_gray_array)

    # mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    # masked_vton_img.save('./images_output/mask.jpg')

    try:
        images = model(
            model_type=model_type,
            category=category_dict[category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )
    except Exception as e:
        print(f"ootd model gen image error: {e}")
        torch.cuda.empty_cache()
        print(f"cuda empty_cache done")
        raise OOTDGenError(e)

    # image_idx = 0
    # for image in images:
    #     image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
    #     image_idx += 1

    # result_path = project_dir / f"./run/test/{uuid.uuid4().hex}.png"
    img: Image.Image = images[0]
    # images[0].save(str(result_path))
    img_io = io.BytesIO()
    img.save(img_io, format="png")
    image_bytes = img_io.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode()

    return image_base64


class GenRequest(BaseModel):
    model: str
    cloth: str
    mask: str


@app.post(path="/process", summary="生成试穿图片")
def process(req: GenRequest):
    try:
        result = gen_trying_image(req.model, req.cloth, req.mask)
    except OOTDGenError as e:
        print("gen trying image error")
        raise HTTPException(status_code=500, detail=f"{e}")

    response = {
        "image": result
    }
    return response

