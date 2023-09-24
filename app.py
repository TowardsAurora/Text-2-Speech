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

            1. Write an utterance to generate,
            2. Select the model to synthesize with
            3. Select the speaker
            4. Hit "Generate" and listen to the result!

            When you select a Model for the first time,
            it will take a little time to download it.
            """
        )
        with gr.Row(variant="panel"):
            text = gr.Textbox(label="Text", placeholder="Insert your article here...")

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
                gets value from `model_name`, loads model,
                re-initializes tts object, gets list of
                speakers that model supports and set them to `speaker`
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
            gets utterance to synthesize from `text` Textbox
            and speaker name from `speaker` dropdown list.
            speaker name might be empty for single-speaker models.
            Synthesizes the waveform and updates `audio` with it.
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
