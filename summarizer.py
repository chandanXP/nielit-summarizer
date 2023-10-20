import numpy as np
import pandas as pd
from abstractive_summarizer import text_regular_expression, get_ranked_sentences_indices, get_paraphrased_paragraph
from target_text import speech
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")


def process_summary(text, model, tokenizer, csv_file, column_name):
    if not isinstance(text, str):
        # Handle NaN or non-string values
        print(f"Skipped processing an invalid value in '{column_name}' column.")
        return

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
    # Read data from an Excel file into a Pandas DataFrame
    excel_file = 'metaverse-summary.xlsx'  # Replace with the path to your Excel file
    df = pd.read_excel(excel_file)

    # Extract the values from a specific column in the DataFrame
    column_name = 'content'  # Replace with the actual column name
    texts = df[column_name].tolist()

    for text in texts:
        process_summary(text, model, tokenizer, 'final_summary.csv', 'text_summ')


if __name__ == "__main__":
    main()
