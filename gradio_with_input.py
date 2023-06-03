import gradio as gr

from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_chapter_init, parse_instructions, get_content_between_a_b
import hashlib

_CACHE = {}

# Build the semantic search model
embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

def parse_novel_input(novel_input):
    if not novel_input:
        print("ERROR - novel_input is empty")
        return None
    
    novel_settings = {
        "name":"",
        "type":"",
        "description":"",
        "background":"",
        "examples":[]
    }
    novel_settings['name'] = get_content_between_a_b('<NOVEL_NAME>:','<NOVEL_TYPE>',novel_input)
    novel_settings['type'] = get_content_between_a_b('<NOVEL_TYPE>:','<NOVEL_DESCRIPTION>',novel_input)
    novel_settings['description'] = get_content_between_a_b('<NOVEL_DESCRIPTION>:','<NOVEL_BACKGROUND>',novel_input)
    novel_settings['background'] = get_content_between_a_b('<NOVEL_BACKGROUND>:','<NOVEL_EXAMPLES>',novel_input)
    exampleStr = get_content_between_a_b('<NOVEL_EXAMPLES>:','<END OF NOVEL DEFINITION>',novel_input)
    novel_settings['examples'] = [x.strip() for x in exampleStr.split('<START>') if x]

    if novel_settings['description'] != "":
        novel_settings['description'] = "about " + novel_settings['description']

    return novel_settings

def init_prompt(novel_input):
    novel_settings = parse_novel_input(novel_input)
    novel_name = novel_settings['name']
    promptStart = f"""
Please write a {novel_settings['type']} novel {novel_settings['description']} with multiple chapters. 
The name of the novel is: "{novel_name}".
The background and character set of the novel:
{novel_settings['background']}

"""
    exampleStr = ""
    for i, example in enumerate(novel_settings['examples']):
        exampleStr += "Example Paragraph " + str(i+1) + ":\n"
        exampleStr += example + "\n"
    promptExamples = f"""
Please write in similar writing style of the following example paragraphs:
{exampleStr}
"""
    promptEnd = f"""
Follow the format below precisely:
- Write a name of Chapter 1 and a concise outline for Chapter 1 based on the provided background, character set and the example paragraphs.
- Write the first 3 paragraphs of the novel based on the example paragraphs and your outline. Write in novelistic style of example paragraphs and take your time to set the scene.
- Write a summary that captures the key information of the 3 paragraphs.
- Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
Chapter 1: <name of Chapter 1>
Outline:
<content of outline for Chapter 1>
Paragraph 1:
<content for paragraph 1>
Paragraph 2:
<content for paragraph 2>
Paragraph 3:
<content for paragraph 3>
Summary:
<content of summary>
Instruction 1: <content for instruction 1>, be concise but interesting.
Instruction 2: <content for instruction 2>, be concise but interesting.
Instruction 3: <content for instruction 3>, be concise but interesting.

Make sure to be precise and follow the output format strictly.
"""
    return promptStart + promptExamples + promptEnd

def init(novel_input, request: gr.Request):
    global _CACHE
    cookie = request.headers['cookie'].split('; _gat_gtag')[0]
    cookie = hashlib.md5(cookie.encode('utf-8')).hexdigest()

    prompt = init_prompt(novel_input)
    print(f"Init Prompt:\n {prompt}")

    # prepare first init
    init_paragraphs = get_chapter_init(prompt)
    # print(init_paragraphs)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']]),
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }
    
    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    cache = _CACHE[cookie]

    all_paragraphs = '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']])
    written_paras = f"""Chapter: {init_paragraphs['Chapter name']}

Outline: {init_paragraphs['Outline']}

Paragraphs:

{all_paragraphs}"""
    
    long_memory_array = [init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']]
    long_memory = parse_instructions(long_memory_array)

    # RecurrentGPT's input is always the last generated paragraph
    writer_start_input = {
            "output_paragraph": init_paragraphs['Paragraph 3'],
            "output_instruction": None,
    }

    # Init GPT writer and cache
    writer = RecurrentGPT(input=writer_start_input, short_memory=init_paragraphs['Summary'], long_memory=long_memory_array, memory_index=None, embedder=embedder, auto_generate=False)
    cache["swriter"] = writer

    # short memory, long memory, current written paragraphs, 3 next instructions
    return prompt, init_paragraphs['Summary'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(short_memory, long_memory, instruction1, instruction2, instruction3, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))

    cookie = request.headers['cookie'].split('; _gat_gtag')[0]
    cookie = hashlib.md5(cookie.encode('utf-8')).hexdigest()
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = [
            instruction1, instruction2, instruction3]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']


def controled_step(short_memory, long_memory, selected_instruction, current_paras, request: gr.Request):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    cookie = request.headers['cookie'].split('; _gat_gtag')[0]
    cookie = hashlib.md5(cookie.encode('utf-8')).hexdigest()
    cache = _CACHE[cookie]
    if "swriter" not in cache:
        print("ERROR - swriter should exist")
        return "", "", "", "", "", ""
    else:
        writer:RecurrentGPT = cache["swriter"] 

        writer.short_memory = short_memory
        writer.input["output_instruction"] = selected_instruction
        writer.step()

    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], parse_instructions(writer.long_memory), current_paras + '\n\n' + writer.output["output_paragraph"], *writer.output['output_instruction']


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # RecurrentGPT
    Interactive Generation of (Arbitrarily) Long Texts with Human-in-the-Loop
    """)

    with gr.Tab("Human-in-the-Loop"):
        with gr.Row():
            with gr.Column():
                novel_input = gr.Textbox(
                    label="Novel Background & Character Set", max_lines=30, lines=30)

                btn_init = gr.Button(
                    "Generate & Send Init Prompt", variant="primary")
                
                novel_init_prompt = gr.Textbox(
                    label="Init Prompts (Generated)", max_lines=30, lines=30)

            with gr.Column():
                written_paras = gr.Textbox(
                    label="Written Paragraphs (Generated)", max_lines=25, lines=25)
                with gr.Box():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (editable)", max_lines=5, lines=5)
                    long_memory = gr.Textbox(
                        label="Long-Term Memory (Generated)", max_lines=6, lines=6)
                with gr.Box():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="Instruction 1", max_lines=6, lines=6, interactive=False)
                        instruction2 = gr.Textbox(
                            label="Instruction 2", max_lines=6, lines=6, interactive=False)
                        instruction3 = gr.Textbox(
                            label="Instruction 3", max_lines=6, lines=6, interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["Instruction 1", "Instruction 2", "Instruction 3"], label="Instruction Selection",)
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="Selected Instruction (editable)", max_lines=5, lines=5)

                btn_step = gr.Button("Next Step", variant="primary")

        btn_init.click(init, inputs=[novel_input], outputs=[
            novel_init_prompt, short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])

    with gr.Tab("Auto-Generation"):
        with gr.Row():
            with gr.Column():
                novel_input = gr.Textbox(
                    label="Novel Background & Character Set", max_lines=30, lines=30)

                btn_init = gr.Button(
                    "Generate & Send Init Prompt", variant="primary")
                
                novel_init_prompt = gr.Textbox(
                    label="Init Prompts (Generated)", max_lines=30, lines=30)
            with gr.Column():
                written_paras = gr.Textbox(
                    label="Written Paragraphs (Generated)", max_lines=25, lines=25)

                with gr.Box():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (editable)", max_lines=5, lines=5)
                    long_memory = gr.Textbox(
                        label="Long-Term Memory (Generated)", max_lines=6, lines=6)

                with gr.Box():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="Instruction 1 (editable)", max_lines=6, lines=6)
                        instruction2 = gr.Textbox(
                            label="Instruction 2 (editable)", max_lines=6, lines=6)
                        instruction3 = gr.Textbox(
                            label="Instruction 3 (editable)", max_lines=6, lines=6)
                    selected_plan = gr.Textbox(
                        label="Revised Instruction (from last step)", max_lines=3, lines=3)

                btn_step = gr.Button("Next Step", variant="primary")

        btn_init.click(init, inputs=[novel_input], outputs=[
            novel_init_prompt, short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(step, inputs=[short_memory, long_memory, instruction1, instruction2, instruction3, written_paras], outputs=[
            short_memory, long_memory, written_paras, selected_plan, instruction1, instruction2, instruction3])


    demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch(server_port=8006, share=False,
                server_name="0.0.0.0", show_api=False)