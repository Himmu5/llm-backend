from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template():
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, respond gracefully, indicating your willingness to learn, "
    "and keep your answer concise (maximum three sentences)."

    "\n\n"
    "{context}"
)


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ])
    return prompt