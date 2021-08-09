import sys, pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from data_helper import Vocabulary
from argparser import model_opts

sys.path.append("..")


def load_data(args):
	vocab = Vocabulary()
	vocab.load(args.vocab_file, keep_words=args.vocab_size)
	
	df_train = pd.read_csv(args.train_data_file, sep="\t")
	df_train.fillna(value="", inplace=True)
	print("train:", df_train.shape)
	
	df_dev = pd.read_csv(args.dev_data_file, sep="\t")
	df_dev.fillna(value="", inplace=True)
	print("dev:", df_dev.shape)
	
	df_test = pd.read_csv(args.test_data_file, sep="\t")
	df_test.fillna(value="", inplace=True)
	print("test:", df_test.shape)
	
	df_train_sim = pd.read_csv(args.train_sim_file, sep="\t")
	df_train_sim.fillna(value="", inplace=True)
	print("train_sim:", df_train_sim.shape)
	
	df_dev_sim = pd.read_csv(args.dev_sim_file, sep="\t")
	df_dev_sim.fillna(value="", inplace=True)
	print("dev_sim:", df_dev_sim.shape)
	
	df_test_sim = pd.read_csv(args.test_sim_file, sep="\t")
	df_test_sim.fillna(value="", inplace=True)
	print("test_sim:", df_test_sim.shape)
	
	def _do_vectorize(df, name):
		df = df.copy()
		df["sentence"] = df["sentence"].map(eval)
		grouped = df.groupby("doc")
		
		sentence_nums = []
		sentence_cut_words = []
		sentence_word_ids = []
		sentences_lens = []
		roles = []
		query = []
		query_lens = []
		
		for agg_name, agg_df in grouped:
			if len(agg_df) >= args.max_sentence_num:
				sentence_nums.append(args.max_sentence_num)
				roles.append(agg_df["role"][-args.max_sentence_num:])
			else:
				sentence_nums.append(len(agg_df))
				roles.append(agg_df["role"])
			
			tmp_words = []
			last = ""
			for words in agg_df["sentence"]:
				if len(words) <= args.max_sequence_length:
					tmp_words.append(words)
					last = words
				else:
					tmp_words.append(words[:args.max_sequence_length])
					last = words[:args.max_sequence_length]
			
			if len(tmp_words) > args.max_sentence_num:
				tmp_words = tmp_words[-args.max_sentence_num:]
			
			sentences_lens.append([len(x) for x in tmp_words])
			sentence_cut_words.append(tmp_words)
			last_len = len(last)
			query_lens.append(last_len)
			
			word_ids = [vocab.do_encode(x)[0] for x in tmp_words]
			query_ids = vocab.do_encode(last)[0]
			
			if last_len < args.max_sequence_length:
				for i in range(last_len, args.max_sequence_length):
					query_ids.append(0)
			
			word_ids = tf.keras.preprocessing.sequence.pad_sequences(word_ids,
			                                                         maxlen=args.max_sequence_length,
			                                                         padding="post",
			                                                         truncating="post",
			                                                         value=0)

			assert np.max(word_ids) < args.vocab_size
			assert np.max(agg_df["role"]) < 6
			sentence_word_ids.append(word_ids)
			query.append(query_ids)
		
		return sentence_word_ids, roles, sentence_nums, sentences_lens, query, query_lens
	
	def _do_label_vectorize(df):
		df = df.copy()
		df.index = range(len(df))
		df["sentence"] = df["sentence"].map(eval)
		grouped = df.groupby("doc")
		
		decoder_input_word_ids = []
		decoder_output_word_ids = []
		decoder_sentence_lens = []
		
		for agg_name, agg_df in grouped:
			question = {x for x in agg_df["question"]}
			question_text = question.pop()
			# cut_words = eval(question_text)
			cut_words = question_text
			decoder_input_word_ids.append(
				vocab.do_encode(cut_words, mode="bos")[0]
			)
			decoder_output_word_ids.append(
				vocab.do_encode(cut_words, mode="eos")[0]
			)
			decoder_sentence_lens.append(
				len(cut_words) + 1
			)
		return decoder_input_word_ids, decoder_output_word_ids, decoder_sentence_lens
	
	train_sentence_word_ids, train_roles, train_sentence_nums, train_sentences_lens, train_query, train_query_lens = _do_vectorize(
		df_train, name=True)
	dev_sentence_word_ids, dev_roles, dev_sentence_nums, dev_sentences_lens, dev_query, dev_query_lens = _do_vectorize(
		df_dev, name=True)
	test_sentence_word_ids, test_roles, test_sentence_nums, test_sentences_lens, test_query, test_query_lens = _do_vectorize(
		df_test, name=True)
	
	train_similar_word_ids, train_similar_roles, train_similar_nums, train_similar_lens, train_squery, train_squery_lens = _do_vectorize(
		df_train_sim, name=False)
	dev_similar_word_ids, dev_similar_roles, dev_similar_nums, dev_similar_lens, dev_squery, dev_squery_lens = _do_vectorize(
		df_dev_sim, name=False)
	test_similar_word_ids, test_similar_roles, test_similar_nums, test_similar_lens, test_squery, test_squery_lens = _do_vectorize(
		df_test_sim, name=False)
	
	train_decoder_input_word_ids, train_decoder_output_word_ids, train_decoder_sentence_lens = _do_label_vectorize(
		df_train)
	dev_decoder_input_word_ids, dev_decoder_output_word_ids, dev_decoder_sentence_lens = _do_label_vectorize(df_dev)
	test_decoder_input_word_ids, test_decoder_output_word_ids, test_decoder_sentence_lens = _do_label_vectorize(df_test)
	
	# f = open("./data_padd", 'w', encoding="UTF-8")
	with open(args.data_file, 'wb') as pkl_file:
		data = [
			list(zip(
				train_sentence_word_ids, train_roles, train_sentence_nums, train_sentences_lens, train_query,
				train_query_lens,
				train_similar_word_ids, train_similar_roles, train_similar_nums, train_similar_lens,
				train_decoder_input_word_ids, train_decoder_output_word_ids, train_decoder_sentence_lens)),
			list(zip(
				dev_sentence_word_ids, dev_roles, dev_sentence_nums, dev_sentences_lens, dev_query, dev_query_lens,
				dev_similar_word_ids, dev_similar_roles, dev_similar_nums, dev_similar_lens,
				dev_decoder_input_word_ids, dev_decoder_output_word_ids, dev_decoder_sentence_lens)),
			list(zip(
				test_sentence_word_ids, test_roles, test_sentence_nums, test_sentences_lens, test_query,
				test_query_lens,
				test_similar_word_ids, test_similar_roles, test_similar_nums, test_similar_lens,
				test_decoder_input_word_ids, test_decoder_output_word_ids, test_decoder_sentence_lens))
		]
		# f.write(str(data))
		pickle.dump(data, pkl_file)
	return data


if __name__ == '__main__':
	args = model_opts()
	load_data(args)