"""
Copyright 2022 Balacoon

TTS interactive demo
"""

import logging
from typing import cast

import gradio as gr
from balacoon_tts import TTS
from huggingface_hub import hf_hub_download, list_repo_files

# global tts module, initialized from a model selected
tts = None


def main():
    logging.basicConfig(level=logging.INFO)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <h1 align="center">Text-to-Speech</h1>

            <p>1. 写一个语句来生成</p>
            <p>2. 选择要合成的模型</p>
            <p>3. 选择说话者</p>
            <p>4. 点击 "生成 "并聆听结果！</p>
            """
        )
        with gr.Row(variant="panel"):
            text = gr.Textbox(label="Text", placeholder="请输入语句，当前仅支持英文......")

        with gr.Row():
            with gr.Column(variant="panel"):
                repo_files = list_repo_files(repo_id="balacoon/tts")
                model_files = [x for x in repo_files if x.endswith("_cpu.addon")]
                model_name = gr.Dropdown(
                    label="Model",
                    choices=model_files,
                )
            with gr.Column(variant="panel"):
                speaker = gr.Dropdown(label="Speaker", choices=[])

            def set_model(model_name_str: str):
                """
                从 `model_name` 获取值，加载模型、
                重新初始化 tts 对象，获取模型支持的扬声器列表并将其设置为
                扬声器列表，并将其设置为 `speaker
                """
                model_path = hf_hub_download(
                    repo_id="balacoon/tts", filename=model_name_str
                )
                global tts
                tts = TTS(model_path)
                speakers = tts.get_speakers()
                value = speakers[-1]
                return gr.Dropdown.update(
                    choices=speakers, value=value, visible=True
                )

            model_name.change(set_model, inputs=model_name, outputs=speaker)

        with gr.Row(variant="panel"):
            generate = gr.Button("Generate")
        with gr.Row(variant="panel"):
            audio = gr.Audio()

        def synthesize_audio(text_str: str, speaker_str: str = ""):
            """
            从`text`文本框中获取要合成的语句
            和 `speaker` 下拉列表中的说话人名称。
            对于单扬声器模型，扬声器名称可能为空。
            合成波形并更新`audio`。
            """
            if not text_str:
                logging.info("text or speaker are not provided")
                return None
            global tts
            if len(text_str) > 10024:
                text_str = text_str[:10024]
            samples = cast(TTS, tts).synthesize(text_str, speaker_str)
            return gr.Audio.update(value=(cast(TTS, tts).get_sampling_rate(), samples))

        generate.click(synthesize_audio, inputs=[text, speaker], outputs=audio)

    demo.launch()


if __name__ == "__main__":
    main()
