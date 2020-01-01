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
        deps = {}
        for word in sentence:
            if word.dep_ == 'relcl':
                relcl = ' '.join(map(str, list(word.subtree)))
                relcls.append(relcl)
            deps[word.text] = word.dep_
        for relcl in relcls:
            relativizer = '-'
            head_noun = '-'
            rc_noun_rel = ''
            mc_noun_rel = ''
            for x in NLP(relcl):
                if x.dep_ == "nsubj":
                    relativizer = x.text

                if x.pos_ == 'NOUN':
                    head_noun = x.text
                    rc_noun_rel = x.dep_
                    mc_noun_rel = deps[x.text]

            results.append({
                'sent_id': idx + 1,
                'sentence': sentence.text,
                'has_rc': 1 if relcls else 0,
                'RC': '"' + relcl + '"',
                'Relativizer': relativizer,
                'Head Noun': head_noun,
                'Gram. Role in MC': mc_noun_rel,
                'Gram. Role in RC': rc_noun_rel
               })
    return results


def write_to_csv(results, output_filename):
    """
    Write the results to the given file path (.csv)
    """
    fieldnames = ["sent_id", "sentence", "has_rc", "RC", "Relativizer", "Head Noun", "Gram. Role in MC", "Gram. Role in RC"]
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
