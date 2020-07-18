import re
import unicodedata
import nltk
from tqdm import tqdm
from string import digits
from bs4 import BeautifulSoup #Nettoyage d'HTML

digits_list = digits


class CleanText:

    def __init__(self):

        french_stopwords = nltk.corpus.stopwords.words('french')
        self.stopwords = [self.remove_accent(sw) for sw in french_stopwords]

        self.stemmer = nltk.stem.SnowballStemmer('french')

    @staticmethod
    def remove_html_code(txt):
        txt = BeautifulSoup(txt, "html.parser", from_encoding='utf-8').get_text()
        return txt

    @staticmethod
    def convert_text_to_lower_case(txt):
        return txt.lower()

    @staticmethod
    def remove_accent(txt):
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode("utf-8")

    @staticmethod
    def remove_non_letters(txt):
        return re.sub('[^a-z_]', ' ', txt)

    def remove_stopwords(self, txt):
        return [w for w in txt.split() if (w not in self.stopwords)]

    def get_stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def apply_all_transformation(self, txt):
        txt = self.remove_html_code(txt)
        txt = self.convert_text_to_lower_case(txt)
        txt = self.remove_accent(txt)
        txt = self.remove_non_letters(txt)
        tokens = self.remove_stopwords(txt)
        tokens_stem = self.get_stem(tokens)
        return tokens_stem

    def clean_df_column(self, df, column_name, clean_column_name):
        nb_line = df.shape[0]
        print("Start Clean %d lines" % nb_line)
        df[clean_column_name] =[ " ".join(self.apply_all_transformation(x)) for x in tqdm(df[column_name].values)]
