import re
import json
import random
import logging
import requests
from random import choice
from typing import Tuple, Union
from translator import BaiDuTranslator


class BaseChatModel:
    def __init__(self, url) -> None:
        self.URL = url
        ...

    def gen_output(self, *args, **kw_args) -> Tuple[Union[str, dict], list]:
        logging.info(f"sent to {self.URL}")

        rst = requests.post(
            url=self.URL,
            *args,
            **kw_args,
        )
        rst = json.loads(rst.content.decode("utf-8"))

        logging.info(f"received {rst}")

        response = rst.get("response")
        history = rst.get("history")
        return response, history

    def gen_system_info(self, *args, **kargs) -> dict: ...

    def gen_history_list(self, *args, **kargs) -> list: ...

    def gen_query(self, query: str, history_list: list) -> str: ...

    def fix_response(self, response: str) -> str:
        return response


class ClockModel(BaseChatModel):

    def gen_output(self, path: str) -> str:
        rst = requests.post(self.URL, files={"image": open(path, "rb")})
        rst = json.loads(rst.content.decode("utf-8"))
        if rst.get("status") == "success":
            pretext = "The clock in the picture now points to {hour} o'clock {minute} minutes.|Yes, The time displayed on the clock is {hour} o'clock {minute} minutes.".format(
                **rst
            )
        else:
            pretext = ""

        return pretext


class CogVLMModel(BaseChatModel):
    def __init__(self, url, zh2en: BaiDuTranslator, en2zh: BaiDuTranslator) -> None:
        super().__init__(url)
        self.zh2en = zh2en
        self.en2zh = en2zh
        self.text_only_template = "A chat between a curious user and an artificial intelligence assistant, Xiaoqian. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

    def gen_history_list(self, history_list: list, history: str) -> list:
        if history == "":
            return history_list

        history_split = history.split("|")
        for idx, _history in enumerate(history_split):
            _history = self.zh2en.translate(_history)
            if idx % 2 == 0:
                ask = _history
            else:
                response = _history
                history_list.append((ask, response))

        return history_list

    def gen_query(self, query, history_list, template=False) -> str:
        en_query: str = self.zh2en.translate(query)

        pattern = r"[\.\?\!]\s*$"
        if re.search(pattern, en_query):
            ...
        else:
            en_query += "?"

        if template:
            old_prompt = self.text_only_template
            for _, (old_query, response) in enumerate(history_list):
                old_prompt += "USER:{}\nASSISTANT:{}\n".format(old_query, response)
            return old_prompt + "USER:{}\nASSISTANT:".format(en_query)
        else:
            return en_query

    def fix_response(self, response: str) -> str:
        response = response.replace("lays", "乐事")
        response = response.replace("mouse", "鼠标")

        if response == "":
            zh_response = "好的"
        elif len(response) <= 10 and "right" in response.lower():
            zh_response = "在右边"
        elif len(response) <= 10 and "orange" in response.lower():
            zh_response = "橙子"
        elif (
            "answering does not require reading text in the image" in response.lower()
            or "unanswerable" in response.lower()
        ):
            zh_response = random.choice(
                [
                    "我不明白你的意思，可以重复一次吗",
                    "臣妾做不到啊，可以换个问题吗",
                    "这个问题我暂时回答不了，请换一个吧",
                    "我需要思考一下，请稍等片刻……",
                    "思考中，思考失败，换个问题再试试吧",
                ]
            )
        else:
            zh_response = self.en2zh.translate(response)

        zh_response = zh_response.replace("表格", "桌子")

        return zh_response
        return super().fix_response(zh_response)


class ChatGLM3Model(BaseChatModel):
    def gen_system_info(self, history_list: list, system_info: dict) -> str:
        history_list.append(system_info)

    def gen_history_list(self, history_list: list, history: str) -> list:
        history_split = history.split("|")
        for idx, _history in enumerate(history_split):
            if idx % 2 == 0:
                role = "user"
            else:
                role = "assistant"
            history_list.append({"role": role, "content": _history})

        return history_list

    def gen_query(self, query: str, history_list: list) -> str:
        return query
