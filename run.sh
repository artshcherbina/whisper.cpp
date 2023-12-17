#MODEL=models/ggml-large-v3.bin
MODEL=models/ggml-large-v3-q5_0.bin
#MODEL=models/ggml-large-v3-q3_k.bin
#MODEL=models/ggml-large-v3-q2_k.bin

#LANG=en
LANG=ru
#LANG=auto

./stream -m $MODEL --silence-t 500 -l $LANG --pressure-t 500
