from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

# repo_id = "mistralai/Mistral-Nemo-Instruct-2407"
repo_id = "sarvamai/sarvam-2b-v0.5"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=repo_id, padding=True, truncation=True, max_length=512)
question_answerer = pipeline(
    "question-answering", 
    model=repo_id, 
    tokenizer=tokenizer,
    return_tensors='pt'
)

class LLMConfig:
    def __init__(self):
        self.repo_id = repo_id
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.llm = HuggingFacePipeline(pipeline=question_answerer,model_kwargs={"temperature": 0.7, "max_length": 512})

    def llm_call(self):
        print("llm_call")
    