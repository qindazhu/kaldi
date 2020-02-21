#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

. ./path.sh
. ./cmd.sh

stage=9

# GPU device id to use (count from 0).
# you can also set `CUDA_VISIBLE_DEVICES` and set `device_id=0`
device_id=7

nj=10

train_set=train_cleaned
gmm=tri3_cleaned
nnet3_affix=_cleaned_pybind
tree_affix=
tdnn_affix=1c

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp

lang=data/lang_chain # output lang dir
#ali_dir=exp/tri5a_ali  # input alignment dir
#lat_dir=exp/tri5a_lats # input lat dir
#treedir=exp/chain/tri5_tree # output tree dir

tree_dir=exp/chain${nnet3_affix}/tree_bi${tree_affix}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

# You should know how to calculate your model's left/right context **manually**
model_left_context=28
model_right_context=28
egs_left_context=$[$model_left_context + 1]
egs_right_context=$[$model_right_context + 1]
frames_per_eg=150,110,90
frames_per_iter=1500000
minibatch_size=128

num_epochs=6
lr=0.001

hidden_dim=1024
bottleneck_dim=128
prefinal_bottleneck_dim=256
kernel_size_list="3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

# true to save network output as kaldi::CompressedMatrix
# false to save it as kaldi::Matrix<float>
save_nn_output_as_compressed=false

. ./path.sh
. ./cmd.sh

. parse_options.sh

if [[ $stage -le 0 ]]; then
  for datadir in train dev test; do
    dst_dir=data/mfcc_hires/$datadir
    if [[ ! -f $dst_dir/feats.scp ]]; then
      echo "making mfcc features for LF-MMI training"
      utils/copy_data_dir.sh data/$datadir $dst_dir
      steps/make_mfcc.sh \
        --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" \
        --nj $nj \
        $dst_dir || exit 1
      steps/compute_cmvn_stats.sh $dst_dir || exit 1
      utils/fix_data_dir.sh $dst_dir
    else
      echo "$dst_dir/feats.scp already exists."
      echo "kaldi (local/run_tdnn_1b.sh) LF-MMI may have generated it."
      echo "skip $dst_dir"
    fi
  done
fi

if [[ $stage -le 1 ]]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [[ $stage -le 2 ]]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 5000 data/mfcc/train $lang $ali_dir $treedir
fi

if  [[ $stage -le 3 ]]; then
  echo "creating phone language-model"
  "$train_cmd" exp/chain/log/make_phone_lm.log \
    chain-est-phone-lm \
     "ark:gunzip -c $treedir/ali.*.gz | ali-to-phones $treedir/final.mdl ark:- ark:- |" \
     exp/chain/phone_lm.fst || exit 1
fi

if [[ $stage -le 4 ]]; then
  echo "creating denominator FST"
  copy-transition-model $treedir/final.mdl exp/chain/0.trans_mdl
  cp $treedir/tree exp/chain
  "$train_cmd" exp/chain/log/make_den_fst.log \
    chain-make-den-fst exp/chain/tree exp/chain/0.trans_mdl exp/chain/phone_lm.fst \
       exp/chain/den.fst exp/chain/normalization.fst || exit 1
fi

if [[ $stage -le 5 ]]; then
  echo "generating egs"
  steps/nnet3/chain/get_egs.sh \
    --alignment-subsampling-factor 3 \
    --cmd "$train_cmd" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --frame-subsampling-factor 3 \
    --frames-overlap-per-eg 0 \
    --frames-per-eg $frames_per_eg \
    --frames-per-iter $frames_per_iter \
    --generate-egs-scp true \
    --left-context $egs_left_context \
    --left-context-initial -1 \
    --left-tolerance 5 \
    --right-context $egs_right_context \
    --right-context-final -1 \
    --right-tolerance 5 \
    --srand 0 \
    --stage -10 \
    $train_data_dir \
    $dir $lat_dir $dir/egs
fi

feat_dim=$(cat $dir/egs/info/feat_dim)
output_dim=$(cat $dir/egs/info/num_pdfs)

if [[ $stage -le 6 ]]; then
  echo "merging egs"
  mkdir -p $dir/merged_egs
  num_egs=$(ls -1 $dir/egs/cegs*.ark | wc -l)

  run.pl --max-jobs-run $nj JOB=1:$num_egs $dir/merged_egs/log/merge_egs.JOB.log \
    nnet3-chain-shuffle-egs ark:$dir/egs/cegs.JOB.ark ark:- \| \
    nnet3-chain-merge-egs --minibatch-size=$minibatch_size ark:- \
      ark,scp:$dir/merged_egs/cegs.JOB.ark,$dir/merged_egs/cegs.JOB.scp || exit 1

  #rm $dir/egs/cegs.*.ark
fi

if [[ $stage -le 7 ]]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

if [[ $stage -le 8 ]]; then
  echo "training..."

  mkdir -p $dir/train/tensorboard
  train_checkpoint=
  if [[ -f $dir/train/best_model.pt ]]; then
    train_checkpoint=$dir/train/best_model.pt
  fi
  
  num_gpus=8
  real_lr=$(echo "$lr * $[$num_gpus-1]" | bc)
  echo "$real_lr"
  
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$init_method"
  
  queue.pl -q v100.q --gpu $num_gpus JOB=1:$num_gpus $dir/logs/task.JOB.log python3 ./chain/ddp_train_copy.py \
	--bottleneck-dim $bottleneck_dim \
	--checkpoint=${train_checkpoint:-} \
	--device-id $device_id \
	--dir $dir \
	--feat-dim $feat_dim \
	--hidden-dim $hidden_dim \
	--is-training true \
	--kernel-size-list "$kernel_size_list" \
	--log-level $log_level \
	--output-dim $output_dim \
	--prefinal-bottleneck-dim $prefinal_bottleneck_dim \
	--subsampling-factor-list "$subsampling_factor_list" \
	--train.cegs-dir $dir/merged_egs \
	--train.init-method $init_method \
	--train.ddp.world-size $num_gpus \
	--train.den-fst $dir/den.fst \
	--train.egs-left-context $egs_left_context \
	--train.egs-right-context $egs_right_context \
	--train.l2-regularize 5e-5 \
	--train.leaky-hmm-coefficient 0.1 \
	--train.lr $real_lr \
	--train.num-epochs $num_epochs \
	--train.use-ddp true \
	--train.valid-cegs-scp $dir/egs/valid_diagnostic.scp \
	--train.xent-regularize 0.1 || exit 1;
fi

if [[ $stage -le 9 ]]; then
  echo "inference: computing likelihood"
  for x in test dev; do
    mkdir -p $dir/inference/$x
    if [[ -f $dir/inference/$x/nnet_output.scp ]]; then
      echo "$dir/inference/$x/nnet_output.scp already exists! Skip"
    else
      best_epoch=$(cat $dir/best-epoch-info | grep 'best epoch' | awk '{print $NF}')
      inference_checkpoint=$dir/epoch-${best_epoch}.pt
      queue.pl -q v100.q --gpu 1 $dir/logs/inference-${x}.log python3 ./chain/inference.py \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint $inference_checkpoint \
        --device-id $device_id \
        --dir $dir/inference/$x \
        --feat-dim $feat_dim \
        --feats-scp data/${x}_hires/feats.scp \
        --hidden-dim $hidden_dim \
        --is-training false \
        --log-level $log_level \
        --kernel-size-list "$kernel_size_list" \
        --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
        --model-left-context $model_left_context \
        --model-right-context $model_right_context \
        --output-dim $output_dim \
        --save-as-compressed $save_nn_output_as_compressed \
        --subsampling-factor-list "$subsampling_factor_list" || exit 1;
    fi
  done
fi

if [[ $stage -le 10 ]]; then
  echo "decoding"
  for x in test dev; do
    if [[ ! -f $dir/inference/$x/nnet_output.scp ]]; then
      echo "exp/chain/inference/$x/nnet_output.scp does not exist!"
      echo "Please run inference.py first"
      exit 1
    fi
    echo "decoding $x"

    ./local/decode.sh \
      --nj $nj \
      $dir/graph \
      $dir/0.trans_mdl \
      $dir/inference/$x/nnet_output.scp \
      $dir/decode_res/$x
  done
fi

if [[ $stage -le 11 ]]; then
  echo "scoring"

  for x in test dev; do
    ./local/score.sh --cmd "$decode_cmd" \
      data/${x}_hires \
      $dir/graph \
      $dir/decode_res/$x || exit 1
  done

  for x in test dev; do
    head $dir/decode_res/$x/scoring_kaldi/best_*
  done
fi
