import re
import os
import sys
from fastapi import FastAPI, File, UploadFile
from typing import Union, Tuple, List
from PIL import Image
import tempfile
import json
import time
import logging
import random
from hashlib import md5
from chat import CogVLMModel, ChatGLM3Model, ClockModel
from translator import BaiDuTranslator

sys.stdout = open("chat_history.txt", "a")

app = FastAPI()

BF16 = True
QUANT = True

logging.basicConfig(
    filename="log/main.log",
    filemode="a",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO,
)

CLOCK_URL = "http://127.0.0.1:11451/clock"
COGVLM_URL = "http://127.0.0.1:11451/cogvlm"
CHATGLM_URL = "http://127.0.0.1:11451/chatglm"

app_key = "20240307001986514"
app_secret = "CFoBXNaiaxFC_vE0mjO3"
zh_en_translator = BaiDuTranslator(app_key, app_secret, "zh", "en")
en_zh_translator = BaiDuTranslator(app_key, app_secret, "en", "zh")

tools = [
    {
        "name": "describe_image",
        "description": "识别图片，输出英文描述",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"description": "根据图片提问的英文问题，仅限英文输入"}
            },
            "required": ["question"],
        },
    },
]


system_item = {
    "role": "system",
    "content": "你是一个与用户进行对话的机器人，你配备了视觉相机的机器人，你能够直接看到用户，你能够不用调用api独立进行视觉识别，你能够通过摄像头拍摄面前的物体、用户并进行识别，并与用户进行交互，你能够识别用户手势。对话中出现英文时，你要将它转换为中文。你根据问题给出简短的回复，使用简体中文回答，不要输出英文，不要输出观测值，不要输出系统信息。你能够使用以下工具：",
    "tools": tools,
}


async def describe_image(question: str, path: str, history: str) -> str:
    logging.info(f"called describe image with {question}, {path}, {history}")

    if question.isascii():
        query = question
    else:
        query = zh_en_translator(question)

        logging.info(f"translated {question} to {query}")

    requests_param = {"query": query, "history": history}
    requests_files = {"image": open(path, "rb")}

    response, _ = vlm_model.gen_output(params=requests_param, files=requests_files)

    response = response.replace("</s>", "")

    response = json.dumps({"description": response})

    return response


async def time_promopt(path: str) -> List[Tuple[str, str]]:
    logging.info(f"called time promopt with {path}")

    history = []
    pre_promopt = clk_model.gen_output(path)
    return vlm_model.gen_history_list(history, pre_promopt)


async def print_chat_history(*args, **kw_args):
    now = time.strftime("#" * 15 + r" %y-%m-%d %H:%M:%S " + "#" * 15, time.localtime())

    print(now)
    print(*args, sep="\n")
    print(*kw_args.items(), sep="\n")
    sys.stdout.flush()


clk_model = ClockModel(CLOCK_URL)
vlm_model = CogVLMModel(COGVLM_URL, zh_en_translator, en_zh_translator)
glm_model = ChatGLM3Model(CHATGLM_URL)


@app.post("/mllm2")
async def mllm2(image: UploadFile, text: str = ""):
    buffer = await image.read()
    fd, path = tempfile.mkstemp()
    cost_time = 0
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(buffer)
        img = Image.open(path)

        if text == "":
            response = ""
        else:
            st = time.time()

            history = [system_item]

            requests_param = {"query": text, "history": json.dumps(history)}

            response, history = glm_model.gen_output(params=requests_param)

            if isinstance(response, dict):
                logging.info(f"function calling as {response}")

                if response.get("name") == "describe_image":
                    question = response.get("parameters", {}).get("question", text)
                else:
                    question = text
            else:
                question = text

            pretext = await time_promopt(path)

            description = await describe_image(question, path, json.dumps(pretext))

            logging.info(f"get response from cogvlm {description}")

            requests_param = {
                "query": json.dumps(description),
                "history": json.dumps(history),
                "role": "observation",
            }

            response, history = glm_model.gen_output(params=requests_param)

            if response.isascii():
                response = en_zh_translator(response)

            response: str = glm_model.fix_response(response)

            await print_chat_history(
                "=" * 20 + "MLLM2Chat" + "=" * 20,
                *history,
                user=text,
                assistant=response,
            )

            cost_time = time.time() - st
    finally:
        os.remove(path)
    return {
        "width": img.width,
        "height": img.height,
        "respose": response,
        "cost_time": cost_time,
    }


app_key = "20240307001986514"
app_secret = "CFoBXNaiaxFC_vE0mjO3"
zh_en_translator = BaiDuTranslator(app_key, app_secret, "zh", "en")
en_zh_translator = BaiDuTranslator(app_key, app_secret, "en", "zh")


@app.post("/mllm")
async def mllm(image: UploadFile, text: str = ""):
    buffer = await image.read()
    fd, path = tempfile.mkstemp()
    cost_time = 0
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(buffer)
        img = Image.open(path)
        if text == "":
            response = ""
        else:
            st = time.time()

            en_text = vlm_model.gen_query(text, [])

            pretext = await time_promopt(path)

            requests_param = {"query": en_text, "history": json.dumps(pretext)}
            requests_files = {"image": open(path, "rb")}

            response, _ = vlm_model.gen_output(
                params=requests_param, files=requests_files
            )

            zh_response = vlm_model.fix_response(response)

            await print_chat_history(
                "=" * 20 + "MLLM Chat" + "=" * 20,
                user_en=en_text,
                asst_en=response,
                user_zh=text,
                asst_zh=zh_response,
            )

            cost_time = time.time() - st
    finally:
        os.remove(path)
    return {
        # "log": {"user_en":en_text,  "asst_en":response, "user_zh":text, "asst_zh":zh_response},
        "width": img.width,
        "height": img.height,
        "respose": zh_response,
        "cost_time": cost_time,
    }


@app.post("/chat")
async def chat(
    image: Union[UploadFile, None] = None, query: str = "", history: str = ""
):
    pre_prompt = ""
    if image is not None:
        buffer = await image.read()
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(buffer)
        img = Image.open(path)
        img_width, img_height = img.width, img.height

        pre_prompt = clk_model.gen_output(path)
    else:
        path = ""
        img = None
        img_width, img_height = 0, 0
    cost_time = 0

    if path == "":
        model = glm_model
    else:
        model = vlm_model

    try:
        messages_list = []

        model.gen_system_info(messages_list, system_item)

        if pre_prompt != "":
            messages_list = model.gen_history_list(messages_list, pre_prompt)

        if history == "":
            ...
        else:
            messages_list = model.gen_history_list(messages_list, history)

        if path != "":
            in_query = query
        else:
            in_query = model.gen_query(query, messages_list)

        requests_param = {"query": in_query, "history": json.dumps(messages_list)}

        if path != "":
            requests_files = {"image": open(path, "rb")}

        if query == "":
            response = ""
            new_history = history
        else:
            st = time.time()

            if path == "":
                response, messages_list = glm_model.gen_output(params=requests_param)
            else:
                response, messages_list = vlm_model.gen_output(
                    params=requests_param, files=requests_files
                )

            response = model.fix_response(response)

            await print_chat_history(
                "=" * 20 + "Chat Chat" + "=" * 20,
                *messages_list,
                user=query,
                assistant=response,
            )

            if history == "":
                new_history = "{query}|{response}".format(
                    query=query, response=response
                )
            else:
                new_history = "{history}|{query}|{response}".format(
                    history=history, query=query, response=response
                )
            cost_time = time.time() - st
    finally:
        if path != "" and os.path.exists(path):
            os.remove(path)
    return {
        "width": img_width,
        "height": img_height,
        "respose": response,
        "history": new_history,
        "cost_time": cost_time,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...), text: str = ""):
    # for cogagent only
    buffer = await image.read()
    fd, path = tempfile.mkstemp()
    cost_time = 0
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(buffer)
        img = Image.open(path)

        st = time.time()

        w = img.width
        h = img.height

        if text.isascii():
            ...
        else:
            text = zh_en_translator.translate(text)

        requests_param = {
            "query": f"Can you point out {text} in the image and provide the bounding boxes of their location?.",
            "history": "[]",
        }
        requests_files = {"image": open(path, "rb")}

        response, _ = vlm_model.gen_output(params=requests_param, files=requests_files)

        pt = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        logging.warning(response)

        out_bbxs = re.findall(pt, response)

        boxes = []

        for bbx in out_bbxs:
            bbx = [int(i) / 1000 for i in bbx]
            boxes.append(
                [int(w * bbx[0]), int(h * bbx[1]), int(w * bbx[2]), int(h * bbx[3])]
            )

        cost_time = time.time() - st
    finally:
        os.remove(path)
    return {
        "width": w,
        "height": h,
        "respose": text,
        "boxes": json.dumps(boxes),
        "cost_time": cost_time,
    }
