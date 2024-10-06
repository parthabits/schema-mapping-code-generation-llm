
# Code Generation using Mistral-7B-Instruct Model

This project demonstrates the usage of the **Mistral-7B-Instruct-v0.1-GPTQ** model for generating Python code to transform data tables. Given two source data tables and a sample target table, the model generates Python code to create a target table that matches the structure of the sample table.

## Requirements

The following dependencies are required for executing the script:

```bash
pip install pandas transformers auto-gptq
```

Additional packages for optimal performance:

```bash
pip install optimum
pip install git+https://github.com/huggingface/transformers.git@72958fcd3c98a7afdc61f953aa58c544ebda2f79
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

## Data Creation

The script generates three CSV files to serve as the input tables: `source1.csv`, `source2.csv`, and `sample.csv`. These files contain patient information with slight variations in the structure, and the goal is to generate a target table in the same format as the sample table.

```python
source1 = '''PatientID,FirstName,LastName,Gender,Age,DateOfBirth,Phone,Email
101,John,Smith,Male,35,15-07-1987,(555) 123-4567,john.smith@email.com
102,Mary,Johnson,Female,28,22-03-1995,(555) 987-6543,mary.j@email.com
103,David,Williams,Male,45,10-12-1978,(555) 555-5555,david.w@email.com
104,Sarah,Brown,Female,52,05-09-1971,(555) 111-2222,sarah.b@email.com
105,Michael,Davis,Male,30,20-11-1992,(555) 333-4444,michael.d@email.com'''

source2 = '''PatientID,Name,Sex,Age,DOB,Telephone,Email
101,John Smith,M,35,15/07/1987,(555) 123-4567,john.smith@email.com
102,Mary Johnson,F,28,22/03/1995,(555) 987-6543,mary.j@email.com
103,David Williams,M,45,10/12/1978,(555) 555-5555,david.w@email.com
104,Sarah Brown,F,52,05/09/1971,(555) 111-2222,sarah.b@email.com
105,Michael Davis,M,30,20/11/1992,(555) 333-4444,michael.d@email.com'''

sample = '''ID,Full Name,Gender,Age,DOB,Mobile,Email
104,Sarah Brown,Female,52,1971-09-05,555-111-2222,sarah.b@email.com
105,Michael Davis,Male,30,1992-11-20,555-333-4444,michael.d@email.com'''
```

The data is loaded into Pandas DataFrames for processing.

```python
import pandas as pd

source1_df = pd.read_csv("source1.csv")
source2_df = pd.read_csv("source2.csv")
sample_df = pd.read_csv("sample.csv")
```

## Model and Code Generation

We use the **Mistral-7B-Instruct** model to generate Python code that transforms the source tables into the structure of the sample table. The model is loaded using `transformers` and `AutoGPTQ`. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

def get_model():
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer
```

### Prompting the Model

A prompt template is constructed to provide the model with the source tables and the task description.

```python
def predict(model, tokenizer, source1_df=source1_df, source2_df=source2_df, sample_df=sample_df):
    source1_row = source1_df.iloc[:2].to_json()
    source2_row = source2_df.iloc[:2].to_json()
    sample_row = sample_df.iloc[:2].to_json()

    prompt = f'''You are an assistant to generate code.
    You are given three tables: Source1, Source2, and Sample.
    The task is to generate a target table that matches the sample table structure, using transformations where necessary.
    Source 1: {source1_row}
    Source 2: {source2_row}
    Sample: {sample_row}

    Python Code:
    '''

    prompt_template = f'<s>[INST] {prompt} [/INST]'
    return get_prediction(model, tokenizer, prompt_template)
```

### Execution

The script loads the model and generates the Python code based on the provided tables.

```python
model, tokenizer = get_model()
predicted_text = predict(model, tokenizer)
print(predicted_text)
```

## Conclusion

This script uses the **Mistral-7B-Instruct** model to generate Python code for transforming data tables. The generated code handles format transformations such as mobile number formatting and date conversions, ensuring that the target table matches the structure of the sample table.
