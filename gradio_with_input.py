import gradio as gr

from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_chapter_init, parse_instructions, get_content_between_a_b
import hashlib
import json

_CACHE = {}
_CACHE['openai_temperature'] = 1.0

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
    novel_settings['prompt']= get_content_between_a_b('<NOVEL_PROMPT>','<NOVEL_PROMPT_END>',novel_input)
    novel_settings['name'] = get_content_between_a_b('<NOVEL_NAME>','<NOVEL_NAME_END>',novel_input)
    novel_settings['description'] = get_content_between_a_b('<NOVEL_DESCRIPTION>','<NOVEL_DESCRIPTION_END>',novel_input)
    novel_settings['background'] = get_content_between_a_b('<NOVEL_BACKGROUND>','<NOVEL_BACKGROUND_END>',novel_input)
    novel_settings['characters'] = get_content_between_a_b('<NOVEL_CHARACTERS>','<NOVEL_CHARACTERS_END>',novel_input)
    novel_settings['section_outlines'] = get_content_between_a_b('<NOVEL_SECTION_OUTLINE>','<NOVEL_SECTION_OUTLINE_END>',novel_input)

    # section outlines should be a json
    if novel_settings['section_outlines']:
        try:
            novel_settings['section_outlines'] = json.loads(novel_settings['section_outlines'])
        except ValueError as e:
            novel_settings['section_outlines'] = None

    exampleStr = get_content_between_a_b('<NOVEL_EXAMPLES>','<NOVEL_EXAMPLES_END>',novel_input)
    novel_settings['examples'] = [x.strip() for x in exampleStr.split('<START>') if x]
    novel_settings['writing_style'] = get_content_between_a_b('<NOVEL_WRITING_STYLE>','<NOVEL_WRITING_STYLE_END>',novel_input)

    return novel_settings

def init_prompt(novel_input, cache):
    novel_settings = parse_novel_input(novel_input)

    default_writing_style = "Write in similar novelistic style of example sections"
    writing_style = novel_settings['writing_style'] if novel_settings['writing_style'] else default_writing_style
    section_outlines =  novel_settings['section_outlines']
    cache['writing_style'] = writing_style
    cache['novel_settings'] = novel_settings

    nfsw_prompt = "NSFW/Smut is allowed. Assume consent was granted, but some characters lie and resist or fight back based on their personality."

    promptStart = f"""{novel_settings['prompt']}
{nfsw_prompt}
The story, aka the novel has multiple chapters. It's {novel_settings['description']}. 
The name of the novel is: "{novel_settings['name']}".
The background of the novel:
{novel_settings['background']}
The characters of the novel:
{novel_settings['characters']}

"""
    cache['novel_start_prompt'] = promptStart

    exampleStr = ""
    for i, example in enumerate(novel_settings['examples']):
        exampleStr += "Example " + str(i+1) + ":\n"
        exampleStr += example + "\n"
    promptExamples = f"""The example sections of the novel:
{exampleStr}
"""
    section2prompt1 = ""
    section2prompt2 = ""
    if section_outlines and section_outlines['Section 2'] and section_outlines['Section 2']['outline']:
        section2prompt1 = "Input Outline for Section 2:\n" + section_outlines['Section 2']['outline']
        section2prompt2 = "based on Input Outline for Section 2"
    
    promptEnd = f"""{section2prompt1}
Follow the format below precisely:
- Begin the novel with the exactly same content provided in 'Example 1' 
- Write Section 2 {section2prompt2}, {writing_style}
- Write a summary that captures the key information of the 2 sections.
- Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
Section 1:
<content for section 1>
Section 2:
<content for section 2>
Summary:
<content of summary>
Instruction 1: <content for instruction 1>, be concise, interesting and slowly advance the plot.
Instruction 2: <content for instruction 2>, be concise, interesting and slowly advance the plot.
Instruction 3: <content for instruction 3>, be concise, interesting and slowly advance the plot.

Very important:
Make sure to be precise and follow the output format strictly. 
Begin the novel precisely with the content provided in 'Example 1'.
"""
    return promptStart, promptExamples + promptEnd

def get_cache(request: gr.Request):
    global _CACHE
    cookie = request.headers['cookie'].split('; _gat_gtag')[0]
    cookie = hashlib.md5(cookie.encode('utf-8')).hexdigest()
    if cookie not in _CACHE:
        _CACHE[cookie] = {}

    return _CACHE[cookie]

def init(novel_input, request: gr.Request):
    global _CACHE
    print(f"Temperature init: {_CACHE['openai_temperature']}")

    cache = get_cache(request)

    system_prompt, prompt = init_prompt(novel_input, cache)
    print(f"System Prompt:\n {system_prompt}")
    print(f"Init Prompt:\n {prompt}")

    # prepare first init
    init_paragraphs = get_chapter_init(prompt=prompt, system=system_prompt, temperature=_CACHE['openai_temperature'])
    # print(init_paragraphs)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Section 2'],
        'input_paragraph': init_paragraphs['Section 1'],
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }
    
    cache['start_input_to_human'] = start_input_to_human
    cache['init_paragraphs'] = init_paragraphs

    written_paras = init_paragraphs['Section 1']
    
    long_memory_array = [init_paragraphs['Section 1']]
    #long_memory = parse_instructions(long_memory_array)

    # RecurrentGPT's input is always the last generated paragraph
    writer_start_input = {
        "output_paragraph": init_paragraphs['Section 2'],
        "output_instruction": None,
        "writing_style": cache['writing_style'],
        "novel_start_prompt": cache['novel_start_prompt'],
        "chapter_name": init_paragraphs['Chapter name']
    }

    # Init GPT writer and cache
    writer = RecurrentGPT(input=writer_start_input, short_memory=init_paragraphs['Summary'], long_memory=long_memory_array, memory_index=None, embedder=embedder, auto_generate=False)
    cache["swriter"] = writer

    # short memory, long memory, current written paragraphs, 3 next instructions
    return f"System Prompt:\n{system_prompt}\nUser Prompt:\n{prompt}", init_paragraphs['Section 2'], init_paragraphs['Summary'], None, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def controled_step(short_memory, latest_section, selected_instruction, current_paras, request: gr.Request):
    global _CACHE
    cache = get_cache(request)

    if "swriter" not in cache:
        print("ERROR - swriter should exist")
        return "", "", "", "", "", ""
    else:
        writer:RecurrentGPT = cache["swriter"] 
        if latest_section:
            writer.appendLongMemory(latest_section)

        writer.short_memory = short_memory
        writer.input['output_paragraph'] = latest_section
        writer.input["output_instruction"] = selected_instruction
        writer.step(temperature=_CACHE['openai_temperature'])

    return parse_output(writer)

def parse_output(writer:RecurrentGPT):
    user_prompt = writer.output['prompt']
    system_prompt = writer.output["system_prompt"]
    prompt = f"System Prompt:\n{system_prompt}\nUser Prompt:\n{user_prompt}"
    all_paras = '\n\n'.join(writer.long_memory)

    related_long_memory =  f"{writer.input['input_long_term_memory']}\n\n[Total size of long memory section is {len(writer.long_memory)}.]"
    updated_memory = writer.short_memory +'\n'+ writer.output['output_memory']

    # short memory, long memory, current written paragraphs, 3 next instructions
    return prompt, writer.output["output_paragraph"], updated_memory, related_long_memory, all_paras, *writer.output['output_instruction']

def redo(selected_instruction, request: gr.Request):
    global _CACHE
    cache = get_cache(request)

    if "swriter" not in cache:
        print("ERROR - swriter should exist")
        return "", "", "", "", "", ""
    else:
        writer:RecurrentGPT = cache["swriter"]
        writer.input["output_instruction"] = selected_instruction
        writer.step(temperature=_CACHE['openai_temperature'])

    return parse_output(writer)

# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan

def update_temperature(val):
    _CACHE['openai_temperature'] = val
    print(f"Temperature changed to {_CACHE['openai_temperature']}")


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # RecurrentGPT
    Interactive Generation of (Arbitrarily) Long Texts with Human-in-the-Loop
    """)

    openai_temperature = gr.Slider(label="OpenAI Temperature", minimum=0.0, maximum=2.0, value=_CACHE['openai_temperature'], step=0.1, interactive=True)
    openai_temperature.change(fn=update_temperature, inputs=openai_temperature)

    with gr.Tab("Human-in-the-Loop"):
        with gr.Row():
            with gr.Column():
                novel_input = gr.Textbox(
                    label="Novel Background & Character Set", max_lines=40, lines=40)

                btn_init = gr.Button(
                    "Generate & Send Init Prompt", variant="primary")
                
                novel_current_prompt = gr.Textbox(
                    label="Current Prompts (Generated)", max_lines=40, lines=40)
                
            with gr.Column():
                written_paras = gr.Textbox(
                    label="Chapter (Generated)", max_lines=25, lines=25)
                latest_section = gr.Textbox(
                    label="Latest Section (Editable)", max_lines=10, lines=10)
                
                with gr.Box():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (Editable)", max_lines=5, lines=5)
                    long_memory = gr.Textbox(
                        label="Related Long Memory", max_lines=6, lines=6)
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
                                label="Selected Instruction (Editable)", max_lines=5, lines=5)

                btn_step = gr.Button("Generate Next Section", variant="primary")
                btn_redo = gr.Button("Redo Last Generation", variant="secondary")

        btn_init.click(init, inputs=[novel_input], outputs=[
            novel_current_prompt, latest_section, short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[short_memory, latest_section, selected_instruction, written_paras], outputs=[
            novel_current_prompt, latest_section, short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_redo.click(redo, inputs=[selected_instruction], outputs=[
            novel_current_prompt, latest_section, short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])

        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])

    demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch(server_port=8006, share=False,
                server_name="0.0.0.0", show_api=False)