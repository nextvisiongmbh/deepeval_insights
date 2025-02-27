from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from EvaluationModel import EvaluationModel
import argparse

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("name")
parser.add_argument("path")
args=parser.parse_args()

def test_correctness(name, path, key):
    
    if key:
        model = 'gpt-4o-mini'
    else:
        model = EvaluationModel(name, path)
    
    correctness_metric = GEval(
        name="Correctness",
        model=model,
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        verbose_mode=True
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output of your LLM application
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    print(test_case)
    assert_test(test_case, [correctness_metric])
    
if __name__ == '__main__':
    name = args.name
    path = args.path
    test_correctness(name, path)