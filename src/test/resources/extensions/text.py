from common import *

from sklearn2pmml.feature_extraction.text import Matcher, Splitter

sentiment_X, sentiment_y = load_sentiment("Sentiment")

stop_words = ["a", "and", "are", "d", "i", "is", "it", "ll", "m", "s", "the", "ve", "we", "you"]

def tokenize(tokenizer, name):
	def process(line):
		tokens = tokenizer(line.lower())
		tokens = [token for token in tokens if token not in stop_words]
		return "\t".join(tokens)
	sentiment_processed_X = sentiment_X.apply(process)
	store_csv(sentiment_processed_X, name)

tokenize(Matcher("(?u)\\b\\w\\w+\\b"), "CountVectorizerSentiment");

tokenize(Matcher("\\w+"), "MatcherSentiment")
tokenize(Splitter("\\s+"), "SplitterSentiment")