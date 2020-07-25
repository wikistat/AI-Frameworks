from clean import CleanText
ct = CleanText()
def apply_all_transformation(txt):
    txt = ct.remove_html_code(txt)
    txt = ct.convert_text_to_lower_case(txt)
    txt = ct.remove_accent(txt)
    txt = ct.remove_non_letters(txt)
    tokens = ct.remove_stopwords(txt)
    tokens_stem = ct.get_stem(tokens)
    return tokens_stem