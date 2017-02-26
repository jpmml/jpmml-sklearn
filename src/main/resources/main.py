from pandas import Series
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

english_series = Series(list(ENGLISH_STOP_WORDS))
english_series.sort_values(inplace = True)
english_series.to_csv("stop_words/english.txt", index = False)
