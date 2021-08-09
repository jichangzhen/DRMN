import datetime, os, sys, logging, nltk, pickle
import tensorflow as tf
import numpy as np
import rouge.rouge_score as rouge_score
from tqdm import tqdm
from collections import deque
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import SmoothingFunction
from mermodel import Model
from optimization import create_optimizer
from argparser import model_opts
from data_helper import WordTable

sys.path.append("..")

def load_data_dict(file):
	def load_data(file):
		with open(file, 'rb') as pkl_file:
			return pickle.load(pkl_file)
	
	train_data, valid_data, test_data = load_data(file)
	_, train_handout_data = train_test_split(train_data, test_size=0.05, random_state=2019)
	print("train_data %d, train_data_handout %d, valid_data %d, test_data %d." % (
		len(train_data), len(train_handout_data), len(valid_data), len(test_data)))
	
	data_dict = {
		"train_data_set": train_data,
		"train_data_set_handout": train_handout_data,
		"valid_data_set": valid_data,
		"test_data_set": test_data
	}
	return data_dict


def cal_rouge(hyp, ref):
	rouge_metrics = {
		"rouge-1": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 1),
		"rouge-2": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 2),
		"rouge-3": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 3),
		"rouge-l": lambda hyp, ref:
		rouge_score.rouge_l_summary_level(hyp, ref),
	}
	scores = {}
	for k, fn in rouge_metrics.items():
		scores[k] = fn(hyp, ref)
	return scores


def tokenid_to_sentenceid(tokenids):
	tokenids = [x for x in tokenids if x not in (0, 1)]
	min_eos_index = tokenids.index(2) if 2 in tokenids else -1
	
	if min_eos_index > 0:
		tokenids = tokenids[:min_eos_index]
	return tokenids


def padding_batch(data_set, num_epochs, batch_size, mode='train'):
	if mode == 'train':
		tf_train_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
			tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
			tf.int32, tf.int32, tf.int32)). \
			shuffle(len(data_set)).repeat(num_epochs).padded_batch(batch_size, padded_shapes=(
			tf.TensorShape([None, None]),
			tf.TensorShape([None]),
			tf.TensorShape([]),
			tf.TensorShape([None]),
			tf.TensorShape([None]),
			tf.TensorShape([]),
			tf.TensorShape([None, None]),
			tf.TensorShape([None]),
			tf.TensorShape([]),
			tf.TensorShape([None]),
			tf.TensorShape([None]),
			tf.TensorShape([None]),
			tf.TensorShape([])),
		                                                           padding_values=(
		                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
		train_iterator = tf_train_data_set.make_one_shot_iterator()
		one_batch = train_iterator.get_next()
	
	else:
		tf_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
			tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
			tf.int32, tf.int32, tf.int32)). \
			padded_batch(batch_size, padded_shapes=(
			tf.TensorShape([None, None]),
			tf.TensorShape([None]),
			tf.TensorShape([]),
			tf.TensorShape([None]),
			tf.TensorShape([None]),
			tf.TensorShape([]),
			tf.TensorShape([None, None]),
			tf.TensorShape([None]),
			tf.TensorShape([]),
			tf.TensorShape([None]),
			tf.TensorShape([None]),
			tf.TensorShape([None]),
			tf.TensorShape([])),
		                 padding_values=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
		valid_iterator = tf_data_set.make_one_shot_iterator()
		one_batch = valid_iterator.get_next()
	return one_batch


def one_step(session, one_batch, model, version, max_decoder_steps, dropout_keep_prob, train=True):
	input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, query_batch, query_len_batch, \
	similar_batch, similar_role_batch, similar_lens_batch, similar_sentences_lens_batch, \
	decoder_input_x_batch, decoder_output_x_batch, decoder_lens_batch = session.run(one_batch)
	if version != 1:
		decoder_input_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_x_batch,
		                                                                      maxlen=max_decoder_steps,
		                                                                      padding="post",
		                                                                      truncating="post",
		                                                                      value=0)
		decoder_output_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_x_batch,
		                                                                       maxlen=max_decoder_steps,
		                                                                       padding="post",
		                                                                       truncating="post",
		                                                                       value=0)
	feed_dict = {model.input_x: input_x_batch,
	             model.input_role: input_role_batch,
	             model.input_sample_lens: input_sample_lens_batch,
	             model.input_sentences_lens: input_sentences_lens_batch,
	             model.query: query_batch,
	             model.query_lens: query_len_batch,
	             model.similar_x: similar_batch,
	             model.similar_role: similar_role_batch,
	             model.similar_sample_lens: similar_lens_batch,
	             model.similar_sentences_lens: similar_sentences_lens_batch,
	             model.decoder_inputs: decoder_input_x_batch,
	             model.decoder_outputs: decoder_output_x_batch,
	             model.decoder_lengths: decoder_lens_batch,
	             model.dropout_keep_prob: dropout_keep_prob,
	             model.training: train
	             }
	return feed_dict, decoder_output_x_batch


def evaluation(predict, groundtruth, mode='train'):
	bleu_scores1 = []
	bleu_scores2 = []
	bleu_scores3 = []
	bleu_scores4 = []
	bleu_scores_2 = []
	bleu_scores_3 = []
	bleu_scores_4 = []
	
	rouge1 = []
	rouge2 = []
	rouge3 = []
	rougel = []
	if mode == 'train':
		test_zip = zip(predict.tolist(), groundtruth.tolist())
	else:
		test_zip = zip(predict, groundtruth)
	for p, g in test_zip:
		gs = tokenid_to_sentenceid(g)
		ps = tokenid_to_sentenceid(p)
		gs, ps = list(map(lambda x: str(x), gs)), list(map(lambda x: str(x), ps))
		score1 = nltk.bleu([gs], ps, weights=(1, 0, 0, 0),
		                   smoothing_function=SmoothingFunction().method1)
		score2 = nltk.bleu([gs], ps, weights=(0, 1, 0, 0),
		                   smoothing_function=SmoothingFunction().method1)
		score3 = nltk.bleu([gs], ps, weights=(0, 0, 1, 0),
		                   smoothing_function=SmoothingFunction().method1)
		score4 = nltk.bleu([gs], ps, weights=(0, 0, 0, 1),
		                   smoothing_function=SmoothingFunction().method1)
		score_2 = nltk.bleu([gs], ps, weights=(1 / 2, 1 / 2, 0, 0),
		                    smoothing_function=SmoothingFunction().method1)
		score_3 = nltk.bleu([gs], ps, weights=(1 / 3, 1 / 3, 1 / 3, 0),
		                    smoothing_function=SmoothingFunction().method1)
		score_4 = nltk.bleu([gs], ps, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4),
		                    smoothing_function=SmoothingFunction().method1)
		
		bleu_scores1.append(score1)
		bleu_scores2.append(score2)
		bleu_scores3.append(score3)
		bleu_scores4.append(score4)
		bleu_scores_2.append(score_2)
		bleu_scores_3.append(score_3)
		bleu_scores_4.append(score_4)
		
		r_scores = cal_rouge([" ".join(ps)], [" ".join(gs)])
		
		rouge1.append(r_scores["rouge-1"]["f"])
		rouge2.append(r_scores["rouge-2"]["f"])
		rouge3.append(r_scores["rouge-3"]["f"])
		rougel.append(r_scores["rouge-l"]["f"])
	return bleu_scores1, bleu_scores2, bleu_scores3, bleu_scores4, bleu_scores_2, bleu_scores_3, bleu_scores_4, rouge1, rouge2, rouge3, rougel


def dev_test_step(session, data_set, model, batch_size, max_decoder_steps, version, dropout_keep_prob):
	num_samples = len(data_set)
	div = num_samples % batch_size
	batch_num = num_samples // batch_size + 1 if div != 0 else num_samples // batch_size
	valid_one_batch = padding_batch(data_set, div, batch_size, mode='test')
	losses = []
	decoder_losses = []
	seq2seq_predicts = []
	seq2seq_y_true = []
	for _ in range(batch_num):
		feed_dict, decoder_output_x_batch = one_step(session, valid_one_batch, model, version, max_decoder_steps,
		                                             dropout_keep_prob, train=False)
		
		fetches = [model.loss, model.decoder_loss, model.infer_predicts]
		loss, decoder_loss, batch_seq2seq_predict = session.run(fetches=fetches, feed_dict=feed_dict)
		
		losses.append(loss)
		decoder_losses.append(decoder_loss)
		seq2seq_predicts.extend(batch_seq2seq_predict.tolist())
		seq2seq_y_true.extend(decoder_output_x_batch.tolist())
	
	bleu_scores1, bleu_scores2, bleu_scores3, bleu_scores4, bleu_scores_2, bleu_scores_3, bleu_scores_4, \
	rouge1, rouge2, rouge3, rougel = evaluation(seq2seq_predicts, seq2seq_y_true, mode='test')
	
	mean_loss = np.mean(losses)
	mean_decoder_loss = np.mean(decoder_losses)
	
	bleu_score1 = np.mean(bleu_scores1) * 100
	bleu_score2 = np.mean(bleu_scores2) * 100
	bleu_score3 = np.mean(bleu_scores3) * 100
	bleu_score4 = np.mean(bleu_scores4) * 100
	bleu_score_2 = np.mean(bleu_scores_2) * 100
	bleu_score_3 = np.mean(bleu_scores_3) * 100
	bleu_score_4 = np.mean(bleu_scores_4) * 100
	
	rouge1 = np.mean(rouge1) * 100
	rouge2 = np.mean(rouge2) * 100
	rouge3 = np.mean(rouge3) * 100
	rougel = np.mean(rougel) * 100
	
	logging.info(
		"bleu_score4 {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g}, bleu_score1 {:g}, bleu_score2 {:g}, bleu_score3 {:g}, bleu_score_2 {:g}, bleu_score_3 {:g}, bleu_score_4 {:g}".format(
			bleu_score4, rouge1, rouge2, rouge3, rougel, bleu_score1,
			bleu_score2, bleu_score3,
			bleu_score_2, bleu_score_3, bleu_score_4))
	
	print(
		"num_samples {}, loss {:g}, decoder_loss {:g},\nbleu_score4 {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g}, bleu_score1 {:g}, bleu_score2 {:g}, bleu_score3 {:g}, bleu_score_2 {:g}, bleu_score_3 {:g}, bleu_score_4 {:g} ".format(
			num_samples, mean_loss, mean_decoder_loss,
			bleu_score4, rouge1, rouge2, rouge3, rougel, bleu_score1,
			bleu_score2, bleu_score3,
			bleu_score_2, bleu_score_3, bleu_score_4))
	
	return mean_loss, mean_decoder_loss


def train(args, data):
	train_data_set = data["train_data_set"]
	train_data_set_handout = data["train_data_set_handout"]
	valid_data_set = data["valid_data_set"]
	test_data_set = data["test_data_set"]
	
	train_num_samples = len(train_data_set)
	batch_num = (train_num_samples * args.num_epochs) // args.batch_size + 1
	
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	session_conf.gpu_options.allow_growth = True
	session_conf = tf.ConfigProto()
	session_conf.gpu_options.allow_growth = True
	
	with tf.Session(graph=tf.Graph(), config=session_conf) as session:
		model = Model(args)
		train_op, learning_rate, global_step = create_optimizer(model.loss,
		                                                        args.learning_rate,
		                                                        num_train_steps=batch_num,
		                                                        num_warmup_steps=int(
			                                                        batch_num * args.warm_up_steps_percent),
		                                                        use_tpu=False)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "sigmod", timestamp))
		print("Writing to {}\n".format(out_dir))
		
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		logging.basicConfig(level=logging.DEBUG,
		                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
		                    datefmt='%a, %d %b %Y %H:%M:%S',
		                    filename=os.path.join(checkpoint_dir, "log.txt"),
		                    filemode='w+')
		logging.info(args)
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
		
		if args.continue_training:
			initialize_op = tf.variables_initializer(
				[x for x in tf.global_variables() if x not in tf.trainable_variables()])
			session.run(initialize_op)
			saver.restore(session, args.checkpoint_path)
		else:
			session.run(tf.global_variables_initializer())
		
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
		losses = deque([])
		losses_steps = deque([])
		
		train_one_batch = padding_batch(train_data_set, args.num_epochs, args.batch_size)
		for batch_id in range(batch_num):
			feed_dict, decoder_output_x_batch = one_step(session, train_one_batch, model, args.use_copy_version,
			                                             args.max_decoder_steps, args.dropout_keep_prob, train=True)
			
			fetches = [update_ops, train_op, learning_rate, global_step, model.loss, model.decoder_loss,
			           model.infer_predicts]
			
			_, _, lr, step, loss, decoder_loss, batch_seq2seq_predict = session.run(fetches=fetches,
			                                                                        feed_dict=feed_dict)
			
			time_str = datetime.datetime.now().isoformat()
			
			print("\n{}:step {}, lr {:g}, loss {:g}, decoder_loss {:g}".format(time_str, step, lr, np.mean(loss),
			                                                                   np.mean(decoder_loss)))
			
			current_step = tf.train.global_step(session, global_step)
			current_lr = session.run(learning_rate)
			
			if current_step % args.evaluate_every == 0:
				logging.info("Evaluation: batch_no %d, global_step %d, learning_rate %.5f." % (
					batch_id, current_step, current_lr))
				
				logging.info("Train result:")
				print("Train result:")
				dev_test_step(session, train_data_set_handout, model, args.batch_size, args.max_decoder_steps,
				              args.use_copy_version, args.dropout_keep_prob)
				
				logging.info("dev result:")
				print("dev result:")
				valid_loss, val_dec_loss = dev_test_step(session, valid_data_set, model, args.batch_size,
				                                         args.max_decoder_steps, args.use_copy_version,
				                                         args.dropout_keep_prob)
				
				logging.info("Test result:")
				print("Test result:")
				dev_test_step(session, test_data_set, model, args.batch_size, args.max_decoder_steps,
				              args.use_copy_version, args.dropout_keep_prob)
				logging.info("\n")
				early_stop = False
				if len(losses) < args.num_checkpoints:
					losses.append(valid_loss)
					losses_steps.append(current_step)
				else:
					if losses[0] == min(losses):
						logging.info("early stopping in batch no %d" % batch_id)
						early_stop = True
					else:
						losses.popleft()
						losses.append(valid_loss)
						losses_steps.popleft()
						losses_steps.append(current_step)
				
				if early_stop:
					print(logging.info("early stop, min valid perplexity is %s." % losses))
					print(logging.info("early stop, stopped at step %d." % losses_steps[0]))
					# break
			
			if current_step % args.checkpoint_every == 0:
				path = saver.save(session, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))
		
		logging.info("******************************Final result***************************")
		print("******************************Final result***************************")
		dev_test_step(session, test_data_set, model, args.batch_size, args.max_decoder_steps, args.use_copy_version,
		              args.dropout_keep_prob)


def main(argv=None):
	args = model_opts()
	args.pre_word_embeddings = WordTable(args.word_emb_file, args.embedding_dim, args.vocab_size).embeddings
	data_dict = load_data_dict(args.data_file)
	train(args, data_dict)


if __name__ == '__main__':
	tf.app.run()
