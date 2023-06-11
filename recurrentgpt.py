from utils import get_content_between_a_b, get_api_response
import torch

import random
from torch import Tensor
from sentence_transformers import  util, SentenceTransformer

class RecurrentGPT:

    def __init__(self, input, short_memory, long_memory, memory_index, embedder:SentenceTransformer, auto_generate=True):
        self.auto = auto_generate
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.embedder = embedder
        if self.long_memory and not memory_index:
            self.memory_index:Tensor = self.embedder.encode(
                self.long_memory, convert_to_tensor=True)
        self.output = {}

    def prepare_input(self, new_character_prob=0.2, top_k=2):
        print("Short memory is: "+self.short_memory)
        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]
        writing_style = self.input["writing_style"]

        instruction_embedding = self.embedder.encode(
            input_instruction, convert_to_tensor=True)

        # get the top most similar sections from memory
        print("Long memory size before step: " + str(self.memory_index.size()))
        memory_scores = util.cos_sim(
            instruction_embedding, self.memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx if idx != len(self.long_memory) - 1]

        print(f"memory scores: {memory_scores}")

        # combine the top sections
        # print(f"Top_k section count: {len(top_k_idx)} Related section count: {len(top_k_memory)}")
        input_long_term_memory = '\n'.join(
            [f"Related Section {i+1} :\n" + selected_memory for i, selected_memory in enumerate(top_k_memory)])
        self.input['input_long_term_memory'] = input_long_term_memory

        # randomly decide if a new character should be introduced
        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the instructions."
            print("Trying to introduce a new character in instructions")
        else:
            new_character_prompt = ""

        input_text = f"""Now I give you a Input Summary (a brief summary of previous stories), you should use it to get the key contents of what has been written so that you can keep track of very long context.
I also give you Input Section (the current section of the novel) and Input Instruction (the instructions to write next section), you should use them to write next section of the novel.
Here are the inputs: 
Input Summary:  
{self.short_memory}
Input Section:
{input_paragraph}
Input Instruction:
{input_instruction}
Input Related Sections:
{input_long_term_memory}

I need you to write:
1. Output Section: the next section of the novel in similar writing style of Input Section and Input Related Sections. Make sure to slowly advance the plot. The output section should follow the input instructions.
2. Output Summary: summarize the key information of the Output Section you've written.
3. Output Instructions: instructions of what to write next (after what you have written). You should output 3 different instructions, each is a possible interesting continuation of the story. Each output instruction should contain around 5 sentences. {new_character_prompt}
Now start writing your output by strictly following the output format as below:
Output Section: 
<content of output section>, {writing_style}
Output Summary: 
<content of output summary>, summarize the key information of the Output Section.
Output Instruction 1: <content for instruction 1>, be concise, interesting and slowly advance the plot.
Output Instruction 2: <content for instruction 2>, be concise, interesting and slowly advance the plot.
Output Instruction 3: <content for instruction 3>, be concise, interesting and slowly advance the plot.

Make sure to be precise and follow the output format strictly.
"""
        return input_text

    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b(
                'Output Section:', 'Output Summary:', output)
            # memory_update_reason = get_content_between_a_b(
            #     'Rationale:', 'Updated:', output)            
            output_memory_updated = get_content_between_a_b(
                'Output Summary:', 'Output Instruction 1:', output)
            ins_1 = get_content_between_a_b(
                'Output Instruction 1:', 'Output Instruction 2', output)
            ins_2 = get_content_between_a_b(
                'Output Instruction 2:', 'Output Instruction 3', output)
            lines = output.splitlines()
            # content of Instruction 3 may be in the same line with I3 or in the next line
            if lines[-1] != '\n' and lines[-1].startswith('Output Instruction 3'):
                ins_3 = lines[-1][len("Output Instruction 3:"):]
            elif lines[-1] != '\n':
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,  # feed to human
                "output_paragraph": output_paragraph,
                "output_instruction": [instruction.strip() for instruction in output_instructions]
            }

            return output
        except:
            return None

    def step(self, temperature, response_file=None):
        prompt = self.prepare_input()
        system_prompt = self.input['novel_start_prompt']

        print(prompt+'\n\n')

        response = get_api_response(content=prompt, system=system_prompt, temperature=temperature)

        self.output = self.parse_output(response)
        while self.output == None:
            response = get_api_response(content=prompt, system=system_prompt, temperature=temperature)
            self.output = self.parse_output(response)
        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Writer's output here:\n{response}\n\n")

        # if self.auto:
        #     # for auto-generation, append the input into long memory because current output need to be extended
        #     self.long_memory.append(self.input["output_paragraph"])
        # elif self.output["output_paragraph"]:
        #     # otherwise append current output into long memory
        #     self.long_memory.append(self.output["output_paragraph"])
        #     # and change output to next input
        #     self.input["output_paragraph"] = self.output["output_paragraph"]

        self.output["prompt"] = prompt
        self.output["system_prompt"] = system_prompt

    def appendLongMemory(self, new_paragraph):
        # add new tensor
        new_tensor = self.embedder.encode(new_paragraph, convert_to_tensor=True)
        existing_tensors = list(self.memory_index)
        existing_tensors.append(new_tensor)
        self.memory_index = torch.stack(existing_tensors)
        
        print("Long memory size after append: " + str(self.memory_index.size()))

        # add text
        self.long_memory.append(new_paragraph)


