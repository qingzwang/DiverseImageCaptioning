import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--initialize_retrieval', type=str, default=None,
                    help="""xxxx.pth""")

    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--cider_optimization', type=int, default=0,
                    help='optimize cider?')


    # Model settings
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')


    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed')

    parser.add_argument('--decoding_constraint', type=int, default=0,
                    help='1 if not allowing decoding two same words in a row, 2 if not allowing any word appear twice in a caption')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--seq_per_img', type=int, default=1,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')


    parser.add_argument('--retrieval_reward_weight_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--retrieval_reward_weight_decay_every', type=int, default=15, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--retrieval_reward_weight_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')


    parser.add_argument('--gate_type', type=str, default='softmax',
                    help='sigmoid or softmax.')
    parser.add_argument('--closest_num', type=int, default=10,
                    help='sigmoid or softmax.')
    parser.add_argument('--closest_file', type=str, default='data/closest.pkl',
                    help='Closest_file')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--rank_eval', type=int, default=0,
                    help='Evaluate vse rank')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')

    # vse
    parser.add_argument('--vse_model', type=str, default="None",
                    help='fc, None')
    parser.add_argument('--vse_rnn_type', type=str, default='gru',
                    help='rnn, gru, or lstm')
    parser.add_argument('--vse_margin', default=0.2, type=float,
                    help='Rank loss margin; when margin is -1, it means use binary cross entropy (usually works with MLP).')
    parser.add_argument('--vse_embed_size', default=1024, type=int,
                    help='Dimensionality of the joint embedding.')
    parser.add_argument('--vse_num_layers', default=1, type=int,
                    help='Number of GRU layers.')
    parser.add_argument('--vse_max_violation', default=1, type=int,
                    help='Use max instead of sum in the rank loss.')
    parser.add_argument('--vse_measure', default='cosine',
                    help='Similarity measure used (cosine|order|MLP)')
    parser.add_argument('--vse_use_abs', default=0, type=int,
                    help='Take the absolute value of embedding vectors.')
    parser.add_argument('--vse_no_imgnorm', default=0, type=int,
                    help='Do not normalize the image embeddings.')
    parser.add_argument('--vse_loss_type', default='contrastive', type=str,
                    help='contrastive or pair')
    parser.add_argument('--vse_pool_type', default='last', type=str,
                    help='last, mean, max')

    # retrieval_reward
    parser.add_argument('--retrieval_reward', default='gumbel', type=str,
                    help='gumbel, reinforce, prob')
    parser.add_argument('--retrieval_reward_weight', default=0, type=float,
                    help='gumbel, reinforce')
    parser.add_argument('--only_one_retrieval', default='off', type=str,
                    help='image, caption, only used when optimizing generator')

    parser.add_argument('--share_embed', default=0, type=int,
                    help='Share embed')
    parser.add_argument('--share_fc', default=0, type=int,
                    help='Share fc')
    parser.add_argument('--caption_loss_weight', default=1, type=float,
                    help='Loss weight.')
    parser.add_argument('--vse_loss_weight', default=0, type=float,
                    help='Loss weight.')

    parser.add_argument('--vse_eval_criterion', default='rsum', type=str,
                    help="The criterion to decide which to take: rsum, t2i_ar, i2t_ar, ....")

    parser.add_argument('--reinforce_baseline_type', default='greedy', type=str,
                    help="no, greedy, gt")

    # add by Qingzhong
    parser.add_argument('--XE_weight', default=1, type=float,
                        help='Loss weight.')
    parser.add_argument('--CIDEr_weight', default=1, type=float,
                        help='Loss weight.')
    parser.add_argument('--DISC_weight', default=1, type=float,
                        help='Loss weight.')
    parser.add_argument('--Div_weight', default=1, type=float,
                        help='diversity weight.')
    parser.add_argument('--num_sample_captions', default=5, type=int,
                        help='number of sample captions.')
    parser.add_argument('--diversity_metric', default='', type=str,
                        help='LSA or selfcider.')
    parser.add_argument('--naive_RL', default=1, type=int,
                        help='use naive RL or not')
    parser.add_argument('--self_critical', default=1, type=int,
                        help='use self critical or not')
    parser.add_argument('--dpp_selection', default=1, type=int,
                        help='use dpp or not')
    parser.add_argument('--subset_num', default=2, type=int,
                        help='the size of dpp set')
    parser.add_argument('--retrieval_quality', default=0, type=float,
                        help='retrieval quality for dpp training')


    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args