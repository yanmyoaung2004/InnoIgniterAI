import dspy

class LawQA(dspy.Signature):
            """Answer user questions strictly based on Myanmar cybersecurity law text. 
            Always cite the relevant Article or Section when possible. 
            If not covered, respond with: 'This law does not explicitly cover that case.'"""    
            question = dspy.InputField(desc="User's legal question")
            answer = dspy.OutputField(desc="Legal answer with citations")


class MyanmarLawAgentTool:
    def __init__(self):
        self.law_qa = dspy.Predict(LawQA)
        with open("./data/law/myanmar-electronic-policy.txt", "r", encoding="utf-8") as f:
            self.law_text = f.read()

    def ask(self, query: str) -> str:
        result = self.law_qa(question=f"{query}\n\nLaw Text:\n{self.law_text}")
        return result.answer

