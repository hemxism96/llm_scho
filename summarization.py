from typing import Any, List, Tuple

from langchain import (
    PromptTemplate,
    LLMChain
)



class summarization:
    """
    Summarizer to compress main ideas from long documentation.
    Inspired from langchain Map-Reduce function.
    """
    def __init__(
            self, 
            context_length,
            map_prompt,
            combine_prompt,
            llm
        ):
        self.context_length = context_length
        self.map_prompt = PromptTemplate(template=map_prompt, input_variables=["text"])
        self.combine_prompt = PromptTemplate(template=combine_prompt, input_variables=["text"])
        self.llm = llm


    def prompt_length(self, docs) -> int:
        """A function to check number of tokens

        Args:
            docs (dict): inputs dictionary for combine prompt

        Returns:
            int: token number of prompt
        """
        prompt = self.combine_llm_chain.prompt.format(**docs)
        return self.combine_llm_chain._get_num_tokens(prompt)


    def _get_inputs(self, docs: List[str]) -> List[dict]:
        """A function to reformat inputs for map prompt

        Args:
            docs (List[str]): list of documentation

        Returns:
            List[dict]: list of input dictionaries
        """
        return [{"text": doc} for doc in docs]


    def _collapse(self, docs: List[str]) -> Tuple[dict, List[str]]:
        """collapse documentation inputs and summarize each documantation

        Args:
            docs (List[str]): documentation inputs

        Returns:
            Tuple[dict, List[str]]: 
                combined documentation summaries,
                prompt token number of summarized documentation
        """
        inputs = self._get_inputs(docs)
        res = [self.map_llm_chain.predict(**input) for input in inputs]
        map_docs =  "\n\n".join(res)
        return {"text": map_docs}, res


    def ReducedMapChain(self, docs: List[str]) -> Any:
        """Repeat summarization until prompt lenght is smaller than llm context length

        Args:
            docs (List[str]): initial documentation inputs

        Returns:
            Any: finial summary of all documentation
        """
        reduced_docs = docs
        while docs:
            reduced_docs_dict, reduced_docs = self._collapse(reduced_docs)
            if self.prompt_length(reduced_docs_dict) < self.context_length:
                break
        return self.combine_llm_chain.predict(**reduced_docs_dict)


    def __call__(self, docs: List[str]):
        self.map_llm_chain = LLMChain(prompt=self.map_prompt, llm=self.llm)
        self.combine_llm_chain = LLMChain(prompt=self.combine_prompt, llm=self.llm)

        return self.ReducedMapChain(docs)