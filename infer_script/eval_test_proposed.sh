# accelerate launch eval_test_proposed.py --conv_type reconstruction
accelerate launch eval_test_proposed.py --conv_type both --pair_cnts 10
accelerate launch eval_test_proposed.py --conv_type adsr --pair_cnts 10
accelerate launch eval_test_proposed.py --conv_type timbre --pair_cnts 10
