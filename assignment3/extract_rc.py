import csv
import spacy
import pandas as pd
from tqdm import tqdm


NLP = spacy.load("en_core_web_sm")


def extract_relative_clause(sentences):
    """
    """
    results = []
    for idx, sentence in tqdm(enumerate(sentences)):
        relcls = []
        for word in sentence:
            if word.dep_ == 'relcl':
                relcl = '"' + ' '.join(map(str, list(word.subtree))) + '"'
                relcls.append(relcl)

        results.append({
                'sent_id': idx,
                'sentence': sentence.text,
                'has_rc': bool(relcls),
                'RCs': ', '.join(relcls)
            })
    return results


def write_to_csv(results, output_filename):
    """
    Write the results to the given file path (.csv)
    """
    fieldnames = ["sent_id", "sentence", "has_rc", "RCs"]
    with open(output_filename, 'w') as out_file:
        dict_writer = csv.DictWriter(out_file, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input-file', help='Input File')
    parser.add_argument('-O', '--output-file', help='Output file path',
                        default='results.csv')
    args = parser.parse_args()

    # Load text file
    print("\nLoading text...")
    sentences = pd.read_table(args.input_file)
    sentences = map(NLP, [sent
                          for _, _, sent in tqdm(sentences.itertuples())])

    # Extract relative clause
    # results = extract_relative_clause(NLP(text).sents)
    print("\nExtracting relative clauses..")
    results = extract_relative_clause(sentences)

    # Write to csv
    write_to_csv(results, args.output_file)
