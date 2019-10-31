import os
import csv
import spacy


def count_syllables(word):
    """
    Count the number of syllables in a word
    #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    """
    count = 0
    vowels = 'aeiouy'

    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1

    if word.endswith('e'):
        if not word.endswith('le'):
            count -= 1

    if word[0] in vowels:
        count += 1

    if not count:
        count += 1
    return count


def compute_metrics(data_dir):
    """
    Compute metrics for the file in the dataset directory.
    """
    nlp = spacy.load("en_core_web_sm")
    metrics = []

    for text_file in os.listdir(data_dir):

        # Skip if it is not a text file
        file_name, ext = os.path.splitext(text_file)
        if ext != '.txt':
            continue

        # Read the text from the file
        with open(os.path.join(data_dir, text_file)) as tf:
            text = tf.read().lower()

        word_count, sentece_count, syllabel_count = [0] * 3
        for sentence in nlp(str(text)).sents:
            sentece_count += 1
            for word in sentence:

                # Skip punctuation
                if word.is_punct:
                    continue

                word_count += 1
                syllabel_count += count_syllables(word.text)

        words_per_sentence = word_count / sentece_count
        syllables_per_word = syllabel_count / word_count

        # Compute Flesch kincaid measure
        grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

        metrics.append({'File': file_name,
                        'N.Sentences': sentece_count,
                        'N.Words': word_count,
                        'N.Syllables': syllabel_count,
                        'Words.Per.Sentence': words_per_sentence,
                        'Syllables.Per.Word': syllables_per_word,
                        'Grade.Score': grade})
    return metrics


def write_to_csv(metrics, output_filename):
    """
    Write the metrics to the given file path (.csv)
    """
    filednames = ['File', 'N.Sentences', 'N.Words', 'N.Syllables',
                  'Words.Per.Sentence', 'Syllables.Per.Word', 'Grade.Score']
    with open(output_filename, 'w') as out_file:
        dict_writer = csv.DictWriter(out_file, filednames)
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

    metrics = compute_metrics(args.input_dir)
    write_to_csv(metrics, args.output_file)
