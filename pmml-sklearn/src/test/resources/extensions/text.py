from sklearn2pmml.feature_extraction.text import Matcher, Splitter

from common import *

stop_words = ["a", "and", "are", "d", "i", "is", "it", "ll", "m", "s", "the", "ve", "we", "you"]

def tokenize(sentiment_df, tokenizer, name):
	sentiment_X = sentiment_df["Sentence"]

	def process(line):
		tokens = tokenizer(line.lower())
		tokens = [token for token in tokens if token not in stop_words]
		return "\t".join(tokens)
	sentiment_processed_X = sentiment_X.apply(process)
	store_csv(sentiment_processed_X, name)

sentiment_df = load_sentiment("Sentiment")

tokenize(sentiment_df, Matcher("(?u)\\b\\w\\w+\\b"), "CountVectorizerSentiment");

tokenize(sentiment_df, Matcher("\\w+"), "MatcherSentiment")
tokenize(sentiment_df, Splitter("\\s+"), "SplitterSentiment")
