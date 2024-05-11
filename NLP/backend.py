from clockreader import ClockReader

from fastapi import FastAPI, File, UploadFile
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
from typing import Union
from PIL import Image

import tempfile
import logging
import torch
import json
import os


app = FastAPI()

torch.manual_seed(114514)

logging.basicConfig(
    filename="log/backend.log",
    filemode="a",
    format=r"%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt=r"%d-%M-%Y %H:%M:%S",
    level=logging.INFO,
)

BF16 = True
QUANT = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLM_MODEL_PATH = "/mnt/data/chatglm3-6b/"

VLM_TOKEN_PATH = "/mnt/data/vicuna-7b-v1.5/"
VLM_MODEL_PATH = "/mnt/data/cogvlm-chat/"
# VLM_MODEL_PATH = "/mnt/data/cogagent-chat-hf/"

if BF16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

logging.info(
    "========Use torch type as:{} with device:{}========\n".format(torch_type, DEVICE)
)

vlm_tokenizer = LlamaTokenizer.from_pretrained(VLM_TOKEN_PATH)
if QUANT:
    vlm_model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True,
    ).eval()
else:
    vlm_model = (
        AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        .to(DEVICE)
        .eval()
    )

logging.debug(f"loaded CogVLM {VLM_MODEL_PATH} TOKEN {VLM_TOKEN_PATH}")


glm3_tokenizer = AutoTokenizer.from_pretrained(GLM_MODEL_PATH, trust_remote_code=True)
glm3_model = (
    AutoModel.from_pretrained(GLM_MODEL_PATH, trust_remote_code=True)
    .quantize(4)
    .cuda()
    .eval()
)

logging.debug(f"loaded ChatGLM {GLM_MODEL_PATH}")


clockreader = ClockReader()


@app.post("/clock")
async def clock(
    image: UploadFile,
):
    logging.info("clock reader called")
    try:
        buffer = await image.read()
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(buffer)

        logging.info("clock image received")

        clock_time = clockreader(path)

        logging.info(f"Predict clock time: {clock_time}")
    finally:
        if path != "" and os.path.exists(path):
            os.remove(path)

    if clock_time is None:
        clock_time = (-1, -1)
        status = "failure"
    else:
        status = "success"
    return {"status": status, "hour": clock_time[0], "minute": clock_time[1]}


@app.post("/cogvlm")
async def chat(
    query: str = "",
    history: str = "[]",
    image: Union[UploadFile, None] = None,
):
    logging.info("CogVLM called")

    if image is not None:
        buffer = await image.read()
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(buffer)
        image = Image.open(path)

        logging.info("image received")
    else:
        image = None
        path = ""

        logging.info("no image received")

    try:
        history = json.loads(history)
    except Exception as e:
        history = []

        logging.error(f"error from json loading: {e}")

    try:
        if image is None:
            input_by_model = vlm_model.build_conversation_input_ids(
                vlm_tokenizer,
                query=query,
                history=history,
                template_version="base",
            )
        else:
            input_by_model = vlm_model.build_conversation_input_ids(
                vlm_tokenizer, query=query, history=history, images=[image]
            )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
            "images": (
                [[input_by_model["images"][0].to(DEVICE).to(torch_type)]]
                if image is not None
                else None
            ),
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(DEVICE).to(torch_type)]
            ]

        gen_kwargs = {"max_length": 2048, "do_sample": False}
    except Exception as e:
        logging.error(f"error from bulid conversation input: {e}")

    try:
        with torch.no_grad():
            outputs = vlm_model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]

            response = vlm_tokenizer.decode(outputs[0])
        response = response.replace("</s>", "")
    except Exception as e:
        response = "system error"

        logging.error(f"error from response generate: {e}")

    if path != "" and os.path.exists(path):
        os.remove(path)

    return {"response": response, "history": history}


@app.post("/chatglm")
async def chat(
    query: str = "",
    history: str = "[]",
    role="user",
):
    try:
        history = json.loads(history)
    except Exception as e:
        history = []

        logging.error(f"error from json loading: {e}")

    try:
        with torch.no_grad():
            response, history = glm3_model.chat(
                glm3_tokenizer, query, history=history, role=role
            )
    except Exception as e:
        response = "system error"

        logging.error(f"error from response generate: {e}")

    return {"response": response, "history": history}
