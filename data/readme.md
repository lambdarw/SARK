# Dataset
**_🐳The whole data will be released upon publication._**


## Data Collection
Our experiments are conducted on the Natural Questions (NQ) dataset, which anonymized user queries paired with Wikipedia articles. To adapt NQ for the reranking task, we execute a rigorous data construction pipeline consisting of three stages.

## Data Statistics
Here is the Statistic of the preprocessed Dataset. (# is the symbol of `the number of').
| Type | # Sample | Percentage |
| :---: | :---: | :---: |
| Training  | 19238  | 78.92%  |
| Validation  | 2138  | 8.77%  |
| Test  | 3000  | 12.31%  |


## Evaluation
We comprehensively assess the generated data across three dimensions: style transfer strength, content preservation, and Fluency. 

**Style Transfer Strength.** We adopt a Qwen2.5-14B model as a binary style classifier. First, to ensure the reliability of this evaluator, we validated it on the formal test set of raw data, where it achieved an F1-score of 95.23%. Subsequently, we applied it to our generated informal paragraphs. The model classified 99.97% of the generated samples as the informal test, confirming that the generated paragraphs are effectively injected with distinct stylistic expressions.

**Content Preservation.** To verify that the semantic meaning remains invariant during style transfer, we measure the semantic similarity between the source formal text and the generated informal text using BERTScore. The dataset achieves a high similarity of 0.6686, indicating that the core information is well-preserved.

**Fluency.** We assess the linguistic quality of the generated text by calculating the Perplexity (PPL) using a GPT-2. A low PPL score of 33.38 suggests that the generated paragraphs are grammatically fluent.
