from typing import Literal, Optional, TypedDict, ChatPrediction
from llama import Role, Message, Dialog, Llama
import fire
import json
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
"""

class RenameMode(Enum):
    Mapping = "mapping"
    Inline = "inline"

class SummaryMode(Enum):
    Function = "function"
    FunctionName = "function_name"
    SegmentLong = "segment_long"
    segmentShort = "segment_short"

"""
    preprocessing ground truth:
    use intermediate file (with macros inlines etc stripped)

    assembly:
    just use it as is?

    preprocessing pseudocode
    parse out segment data (e.g. global variables, symbols)
    v-tables
    extended topological sort of functions
    use existing methods to determine all possible structures (function call sinks as well as declarations, should be purely procedural)

    find previous work to get size of struct / class, note the role of inheritance in c++!

    challenge 1:
    assembling just some functions, not the whole file?

    multiple optimization levels!

    1. permute according to topological order
    2. check if there is new type
    if true:
        add new type to list
        check if new type is dupe with anything
            if true:
                use the old type;
            else:
                use the new type;
    else:
        find best type in list

    remark: maybe the type list construction should be top-down instead? especially for c++ classes
    and big structs

    feature: show each inference step, allow revert

    idea of operation flow
    """

class CodeLlama:
    def __init__(self, num_params: int, ckpt_dir: str, tokenizer_path: str, temperature: float = 0.7, top_p: float = 0.1, max_seq_len: int = 3584, max_batch_size: int = 4, max_gen_len: Optional[int] = None):
        # optimize parameters
        self.model = generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size)
        
        with f as open('instructions.json'):
            self.instructions: dict = json.load(f)
    
    def prompt(self, op: str, sub_op: str, function: str, context: str = "", type: str = "") -> Dialog:
        mode = ""
        text = ""
        if function:
            mode = "function"
            text = "The function to be modified: {}".format(function)
        elif type:
            mode = "type"
            text = "The function to be modified: {}".format(type)
        if context:
            context = "The extenal context of the {}: {}".format(mode, context)

        return [Message("system", self.instructions[op]["system"]), Message("user", self.instructions[op]["user"][sub_op] + text + context)]

    def query_model(self, queries: List[Dialog], temperature: float = 0.6, top_p: float = 0.9) -> List[ChatPrediction]:
        """
        Sends generic queries to the current CodeLlama Instruct model, returning a tuple of the prompts and results.
        :param query: The requests to send, including role and content formatted in JSON
        """
        # todo: cut function into smaller pieces, see gpt wpre
        return self.model.chat_completion(
        queries,
        temperature=temperature,
        top_p=top_p,
        logprobs=False
        )
    
    # refine is called thereafter, with the relevant content provided!
    # give naming convention from the outside
    def rename(self, function: str, mode: RenameMode, reasoning: bool):
        opname = "rename"
        results = self.query_model([self.propmt(opname, mode, function)])
        if mode == RenameMode.Mapping:
            return json.loads(results)
        # permute through functions in the text, feed to refine, but we do this from the outside
        return results

    """
    Refine decompilation of a function based on calls to it from without.
    call refine recursively
    """
    def refine(self, function: str, context: str):
        opname = "refine"
        return self.query_model([self.propmt(opname, "rename", function)])

    # give naming convention from the outside
    def summarize(self):
        pass

    """
    Cleanup code from cfg, possibly with ancient techniques
    """
    def cleanup(self):
        pass
    
    def infer_field(self):
        pass

    def print_query_results(instruction, results):
        for instruction, result in zip(instructions, results):
            for msg in instruction:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            print("\n==================================\n")


    # -----------------------------------------------------------------------------

    def query_model_async(self, query, cb):
        """
        Function which sends a query to {model} and calls a callback when the response is available.
        :param query: The request to send to {model}
        :param cb: Tu function to which the response will be passed to.
        """
        print(_("Request to {model} sent...").format(model=str(gepetto.config.model)))
        t = threading.Thread(target=self.query_model, args=[query, cb])
        t.start()

    # naive approach: run the algo two times
    # see if we can decompile function-by-function, just add the others as references


    # unsupervised training, but we also need some high quality data