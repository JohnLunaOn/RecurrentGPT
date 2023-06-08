from utils import get_content_between_a_b, get_api_response
import torch

import random

from sentence_transformers import  util


class RecurrentGPT:

    def __init__(self, input, short_memory, long_memory, memory_index, embedder, auto_generate=True):
        self.auto = auto_generate
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.embedder = embedder
        if self.long_memory and not memory_index:
            self.memory_index = self.embedder.encode(
                self.long_memory, convert_to_tensor=True)
        self.output = {}

    def prepare_input(self, new_character_prob=0.2, top_k=2):
        print("Short memory is: "+self.short_memory)
        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]
        writing_style = self.input["writing_style"]
        chapter_name = self.input['chapter_name']

        instruction_embedding = self.embedder.encode(
            input_instruction, convert_to_tensor=True)

        # get the top 3 most similar sections from memory

        memory_scores = util.cos_sim(
            instruction_embedding, self.memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx]
        # combine the top 3 sections
        input_long_term_memory = '\n'.join(
            [f"Related Sections {i+1} :\n" + selected_memory for i, selected_memory in enumerate(top_k_memory)])
        # randomly decide if a new character should be introduced
        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the instructions."
            print("Trying to introduce a new character in instructions")
        else:
            new_character_prompt = ""

        input_text = f"""Current chapter is: {chapter_name}.
Now I give you a memory (a brief summary) of 400 words, you should use it to store the key content of what has been written so that you can keep track of very long context. For each time, I will give you your current memory (a brief summary of previous stories. You should use it to store the key content of what has been written so that you can keep track of very long context), the previously written section, and instructions on what to write in the next section. 
I need you to write:
1. Output Section: the next section of the novel in similar writing style of Input Section and Input Related Sections. The output section should follow the input instructions.
2. Output Memory: compose a summary that encapsulates the pivotal information associated with the Input Memory and the Output Section. Begin by detailing what should be integrated into the Input Memory and provide a justification for these additions. Following this, present the revised version of the Input Memory, reflecting the updates.
3. Output Instruction: instructions of what to write next (after what you have written). You should output 3 different instructions, each is a possible interesting continuation of the story. Each output instruction should contain around 5 sentences. {new_character_prompt}
Here are the inputs: 
Input Memory:  
{self.short_memory}
Input Section:
{input_paragraph}
Input Instruction:
{input_instruction}
Input Related Sections:
{input_long_term_memory}

Now start writing, organize your output by strictly following the output format as below:
Output Section: 
<content of output section>, around 30 - 50 sentences. {writing_style}
Output Memory: 
Rational: <string that explain how to update the memory>
Updated Memory:
<string of updated memory>, around 20 sentences
Output Instruction: 
Instruction 1: <content for instruction 1>, be concise, interesting and slowly advance the plot.
Instruction 2: <content for instruction 2>, be concise, interesting and slowly advance the plot.
Instruction 3: <content for instruction 3>, be concise, interesting and slowly advance the plot.
Very important:
The updated memory should only store key information. You should first explain what needs to be added into or deleted from the memory and why. After that, you start rewrite the input memory to get the updated memory.
Make sure to be precise and follow the output format strictly.
"""
        return input_text

    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b(
                'Output Section:', 'Output Memory', output)
            memory_update_reason = get_content_between_a_b(
                'Rational:', 'Updated Memory:', output)            
            output_memory_updated = get_content_between_a_b(
                'Updated Memory:', 'Output Instruction:', output)
            self.short_memory = output_memory_updated
            ins_1 = get_content_between_a_b(
                'Instruction 1:', 'Instruction 2', output)
            ins_2 = get_content_between_a_b(
                'Instruction 2:', 'Instruction 3', output)
            lines = output.splitlines()
            # content of Instruction 3 may be in the same line with I3 or in the next line
            if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
                ins_3 = lines[-1][len("Instruction 3:"):]
            elif lines[-1] != '\n':
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,  # feed to human
                "memory_update_reason": memory_update_reason,
                "output_paragraph": output_paragraph,
                "output_instruction": [instruction.strip() for instruction in output_instructions]
            }

            return output
        except:
            return None

    def step(self, temperature, response_file=None):

        prompt = self.prepare_input()
        system_prompt = self.input['novel_start_prompt']

        print(prompt+'\n'+'\n')

        response = get_api_response(content=prompt, system=system_prompt, temperature=temperature)

        self.output = self.parse_output(response)
        while self.output == None:
            response = get_api_response(content=prompt, system=system_prompt, temperature=temperature)
            self.output = self.parse_output(response)
        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Writer's output here:\n{response}\n\n")

        if self.auto:
            # for auto-generation, append the input into long memory because current output need to be extended
            self.long_memory.append(self.input["output_paragraph"])
        elif self.output["output_paragraph"]:
            # otherwise append current output into long memory
            self.long_memory.append(self.output["output_paragraph"])
            # and change output to next input
            self.input["output_paragraph"] = self.output["output_paragraph"]

        self.output["prompt"] = prompt
        self.output["system_prompt"] = system_prompt
        self.memory_index = self.embedder.encode(
            self.long_memory, convert_to_tensor=True)
