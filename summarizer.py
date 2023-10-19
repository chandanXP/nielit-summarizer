import numpy as np
import pandas as pd
from abstractive_summarizer import text_regular_expression, get_ranked_sentences_indices, get_paraphrased_paragraph
from target_text import speech
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
import multiprocessing
from functools import partial


model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")


def process_summary(text, model, tokenizer, csv_file, column_name):
    num_sentences = 8
    sentences = text_regular_expression(text)

    n = get_ranked_sentences_indices(sentences, num_sentences)
    extractive_summary = ' '.join(np.array(sentences)[n])

    # You might want to capture the paraphrased result here, if needed
    paraphrased_summary = get_paraphrased_paragraph(model, tokenizer, extractive_summary, num_return_sequences=1, num_beams=5)

    # Append to the CSV file (new row)
    data = {'text_summary': [extractive_summary]}
    df = pd.DataFrame(data)
    df.to_csv(csv_file, mode='a', header=False, index=False)
    print(f'Appended "{extractive_summary}" to {csv_file} in the "{column_name}" column (new row).')


def main():
    # Define your list of string values
    texts = ['text1', 'text2', 'text3', ...]  # Replace with your actual text data

    # Set the number of CPU cores to utilize
    num_cores = multiprocessing.cpu_count()

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(process_summary, model=model, tokenizer=tokenizer, csv_file='final_summary.csv', column_name='text_summary')
        pool.map(func, texts)


if __name__ == "__main__":
    main()
