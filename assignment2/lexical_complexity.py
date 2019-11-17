import os
from collections import defaultdict

import csv
import spacy
import pandas as pd
import numpy as np


def transform_subtelexus():
    return


def iterate_dir(data_dir):
    """
    Iterate dir recursively to read all the files
    """
    for child in os.listdir(data_dir):
        child_path = os.path.join(data_dir, child)
        if os.path.isdir(child_path):
            print(child_path, ":- is a dir")
            yield from iterate_dir(child_path)
        else:
            with open(child_path) as tf:
                text = tf.read()
            yield (child, os.path.split(data_dir)[1], text)


def lexical_complexity(text):
    """
    Compute the lexical complexity measures of the given text
    """
    lexs, tokens, types = [0] * 3
    freq1, freq2 = [], []
    vocab = defaultdict(int)
    for word in text:
        pos = word.pos_
        if pos in ["PUNCT", "SYM", "SPACE"]:
            continue
        tokens += 1

        word_text = word.text.lower()
        vocab[word_text] += 1

        f1 = get_freq(word_text, "Zipf-value")
        f2 = get_freq(word_text, "Lg10WF")
        if f1:
            freq1.append(f1)
            freq2.append(f2)
        # else:
        #    print("Missing |%s| with pos %s" % (word_text, pos))

        if pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
            lexs += 1

    # Compute lexical density
    lexical_density = lexs / tokens

    # Compute term token ratio
    ttr = len(vocab) / tokens

    # Compute mean and standard deviation of Zipf-value
    mean_freq1 = sum(freq1) / len(freq1)
    sd_freq1 = np.std(np.array(freq1))

    # Compute mean and standard deviation of Lg10WF
    mean_freq2 = sum(freq2) / len(freq2)
    sd_freq2 = np.std(np.array(freq2))

    # Stats for debugging:
    # print("Total number of tokens:", tokens)
    # print("Total number of lexs:", lexs)
    # print("Total number of types:", len(vocab))
    # input("Press Enter to continue...")

    return {"Lexical.density": lexical_density,
            "Mean.frequency.Zipf-value": mean_freq1,
            "Mean.frequency.Lg10WF": mean_freq2,
            "Sd.frequency.Zipf-value": sd_freq1,
            "Sd.frequency.Lg10WF": sd_freq2,
            "TTR": ttr}


def get_freq(word, key):
    """
    """
    if word in freq_lut:
        return freq_lut[word][key]
    else:
        None


def write_to_csv(metrics, output_filename):
    """
    Write the metrics to the given file path (.csv)
    """
    fieldnames = ["File", "Reading.level", "Lexical.density",
                  "Mean.frequency.Zipf-value", "Sd.frequency.Zipf-value",
                  "Mean.frequency.Lg10WF", "Sd.frequency.Lg10WF", "TTR"]
    with open(output_filename, 'w') as out_file:
        dict_writer = csv.DictWriter(out_file, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(metrics)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input-dir', help='Input directory')
    parser.add_argument('-O', '--output-file', help='Output file path',
                        default='results.csv')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print("No such directory exists !!!")
        exit()

    # Load model and freq look up table
    nlp = spacy.load("en_core_web_sm")
    subtlexus = pd.read_table("./subtlexus.csv",
                              usecols=['Word', 'Lg10WF', 'Zipf-value'])
    freq_lut = {str(w).lower(): {'Lg10WF': f, 'Zipf-value': z}
                for _, (w, f, z) in subtlexus.iterrows()}

    all_metrics = []
    for file_name, level, text in iterate_dir(args.input_dir):
        print(file_name)
        metrics = lexical_complexity(nlp(text))
        metrics.update({'File': file_name, 'Reading.level': level})
        all_metrics.append(metrics)

    # Write to csv
    write_to_csv(all_metrics, args.output_file)
