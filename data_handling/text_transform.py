from re import sub

def text_preprocess(sentence):
    # transform to lower case
    def clean_text(text):
        text = text.lower()
        # remove any forgotten space before punctuation and double space
        text = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', text).replace('  ', ' ')
        # remove punctuations
        text = sub('[(,.!?;:|*\")]', ' ', text).replace('  ', ' ')
        return text

    if isinstance(sentence, str):
        return clean_text(sentence)
    elif isinstance(sentence, list):
        return [clean_text(s) for s in sentence]
    else:
        raise ValueError("Input should be a string or a list of strings.")
