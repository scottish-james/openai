#  README
## Overview
This project provides a set of tools to process and analyze text data, specifically focusing on embedding generation and similarity calculations using OpenAI's models. The core functionalities include reading data, generating embeddings, computing cosine similarities, and searching for the most relevant answers and questions based on given input.

## Dependencies
Ensure you have the following dependencies installed:

- numpy
- pandas
- openai

You can install them using pip:

```
pip install numpy pandas openai

```

## Usage
### Reading Data
The script reads data from a CSV file (embedded.csv) that contains precomputed embeddings for answers and questions


```python
import numpy as np
import pandas as pd
import openai

try:
    df = pd.read_csv('embedded.csv')
except ValueError as e:
    print(f"Error reading the file: {e}")
else:
    print(df.dtypes)



```

    Question ID            int64
    Question              object
    Answer                object
    answer_embedding      object
    question_embedding    object
    dtype: object
    

## Setting Up OpenAI API
Ensure you have your OpenAI API key stored in a text file (api_key.txt). The script reads this key to initialize the OpenAI client.


```python
with open('api_key.txt', 'r') as file:
    api_key_1 = file.read().strip()

openai.api_key = api_key_1
```

## Data Preparation
Convert the string representations of embeddings back to numpy arrays for further processing.


```python
df['answer_embedding'] = df['answer_embedding'].apply(eval).apply(np.array)
df['question_embedding'] = df['question_embedding'].apply(eval).apply(np.array)
```

## Functions
### get_embedding
Generates an embedding for a given text using a specified model.


```python
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate an embedding for a given text using a specified model.
    """
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding

```

### cosine_similarity
Calculates the cosine similarity between two vectors.


```python
def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(y * y for y in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)

```

### search_reviews_answer
Finds and returns the top N most similar answers based on a question.


```python
def search_reviews_answer(df, question, n=3, pprint=True):
    """
    Find and return the top N most similar answers based on a question.
    """
    embedding = get_embedding(question, model='text-embedding-3-small')
    df['similarities_answers'] = df['answer_embedding'].apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities_answers', ascending=False).head(n)
    return res

```

### search_reviews_question
Finds and returns the top N most similar questions based on an input question.


```python
def search_reviews_question(df, question, n=3, pprint=True):
    """
    Find and return the top N most similar questions based on an input question.
    """
    embedding = get_embedding(question, model='text-embedding-3-small')
    df['similarities_questions'] = df['question_embedding'].apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities_questions', ascending=False).head(n)
    return res

```

## Examples
Finding the top 10 most similar answers to the question "open times":


```python
df_answer = search_reviews_answer(df, 'open times', n=10)
df_answer.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Question ID</th>
      <th>Question</th>
      <th>Answer</th>
      <th>answer_embedding</th>
      <th>question_embedding</th>
      <th>similarities_answers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>What are the branch opening hours?</td>
      <td>Our branches are open from 9 AM to 5 PM, Monda...</td>
      <td>[-0.01136218011379242, 0.0748688355088234, 0.0...</td>
      <td>[-0.03722250834107399, 0.07355938851833344, 0....</td>
      <td>0.299570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>How can I open a checking account?</td>
      <td>You can open a checking account by visiting an...</td>
      <td>[0.010874899104237556, 0.04530753940343857, 0....</td>
      <td>[0.03132950887084007, 0.031158041208982468, 0....</td>
      <td>0.234716</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>How do I close my account?</td>
      <td>To close your account, visit any of our branch...</td>
      <td>[0.04966616630554199, 0.03897934779524803, 0.0...</td>
      <td>[0.04915893077850342, 0.014077764004468918, 0....</td>
      <td>0.187534</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Can I get a loan to buy a car?</td>
      <td>Yes, we offer auto loans with competitive inte...</td>
      <td>[-0.02668355219066143, 0.019259411841630936, 0...</td>
      <td>[-0.002607405884191394, -0.03094622679054737, ...</td>
      <td>0.184700</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Can I get a statement copy?</td>
      <td>Yes, you can request a statement copy through ...</td>
      <td>[0.031010989099740982, -0.023339280858635902, ...</td>
      <td>[0.026227407157421112, -0.020656602457165718, ...</td>
      <td>0.178007</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>How do I update my contact information?</td>
      <td>You can update your contact information throug...</td>
      <td>[0.01906469650566578, -0.014860356226563454, 0...</td>
      <td>[0.02334478124976158, -0.028389401733875275, 0...</td>
      <td>0.169532</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>How can I check my account balance?</td>
      <td>You can check your account balance through our...</td>
      <td>[0.03639216721057892, 0.0075601255521178246, 0...</td>
      <td>[0.05246749520301819, 0.010690983384847641, 0....</td>
      <td>0.169029</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>How do I apply for a personal loan?</td>
      <td>You can apply for a personal loan online throu...</td>
      <td>[-0.0037767095491290092, 0.015247618779540062,...</td>
      <td>[-0.0032004239037632942, -0.002346499124541878...</td>
      <td>0.159263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>What documents are required to open an account?</td>
      <td>To open an account, you need a valid ID, proof...</td>
      <td>[0.0847620889544487, 0.011813902296125889, 0.0...</td>
      <td>[0.038944222033023834, 0.0715121254324913, 0.0...</td>
      <td>0.142831</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Can I set up automatic bill payments?</td>
      <td>Yes, you can set up automatic bill payments th...</td>
      <td>[0.016405565664172173, -0.02754642628133297, -...</td>
      <td>[0.0043916646391153336, -0.03269881010055542, ...</td>
      <td>0.136826</td>
    </tr>
  </tbody>
</table>
</div>



Finding the top 3 most similar questions to the question "open times":


```python
df_question = search_reviews_question(df, 'open times', n=3)
df_question.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Question ID</th>
      <th>Question</th>
      <th>Answer</th>
      <th>answer_embedding</th>
      <th>question_embedding</th>
      <th>similarities_answers</th>
      <th>similarities_questions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>What are the branch opening hours?</td>
      <td>Our branches are open from 9 AM to 5 PM, Monda...</td>
      <td>[-0.01136218011379242, 0.0748688355088234, 0.0...</td>
      <td>[-0.03722250834107399, 0.07355938851833344, 0....</td>
      <td>0.299570</td>
      <td>0.374231</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Can I get a statement copy?</td>
      <td>Yes, you can request a statement copy through ...</td>
      <td>[0.031010989099740982, -0.023339280858635902, ...</td>
      <td>[0.026227407157421112, -0.020656602457165718, ...</td>
      <td>0.178007</td>
      <td>0.211189</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>What documents are required to open an account?</td>
      <td>To open an account, you need a valid ID, proof...</td>
      <td>[0.0847620889544487, 0.011813902296125889, 0.0...</td>
      <td>[0.038944222033023834, 0.0715121254324913, 0.0...</td>
      <td>0.142831</td>
      <td>0.193357</td>
    </tr>
  </tbody>
</table>
</div>



## Notes
- Ensure the 'embedded.csv' file is correctly formatted and contains the necessary embedding columns.
- The OpenAI API key should be kept secure and not exposed in the codebase.
- The embeddings and similarity calculations assume that the embeddings are valid and correctly formatted numpy array


```python

```



```python

```
